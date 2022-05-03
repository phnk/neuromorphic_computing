from brian2 import *
import matplotlib.pyplot as plt

def simple_synapse():
    eqs = """
        dv/dt = (I-v)/tau : 1
        I : 1
        tau: second
    """

    G = NeuronGroup(2, eqs, threshold="v>1", reset="v=0", method="exact")
    G.I = [2, 0]
    G.tau = [10, 100]*ms

    # No sypnases: no post synaptic spikes
    S = Synapses(G, G, on_pre="v_post += 0.2")
    S.connect(i=0, j=1)

    M = StateMonitor(G, "v", record=True)

    run(100*ms)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(M.t/ms, M.v[0], label="Neuron 0")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("v")

    ax2.plot(M.t/ms, M.v[1], label="Neuron 1")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("v")
    plt.show()

def adding_weight():
    eqs = """
        dv/dt = (I-v)/tau : 1
        I : 1
        tau : second
    """

    G = NeuronGroup(3, eqs, threshold="v>1", reset="v=0", method="exact")
    G.I = [2, 0, 0]
    G.tau = [10, 100, 100]*ms

    S = Synapses(G, G, "w:1", on_pre="v_post+=w")
    S.connect(i=0, j=[1,2])
    S.w = "j*0.2"

    M = StateMonitor(G, "v", record=True)

    run(50*ms)

    plt.plot(M.t/ms, M.v[0], label="Neuron 0")
    plt.plot(M.t/ms, M.v[1], label="Neuron 1")
    plt.plot(M.t/ms, M.v[2], label="Neuron 2")
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.legend()
    plt.show()

def introducing_delay():
    eqs = """
        dv/dt = (I-v)/tau : 1
        I : 1
        tau : second
    """

    G = NeuronGroup(3, eqs, threshold="v>1", reset="v=0", method="exact")
    G.I = [2, 0, 0]
    G.tau = [10, 100, 100]*ms

    S = Synapses(G, G, "w:1", on_pre="v_post+=w")
    S.connect(i=0, j=[1, 2])
    S.w = "j*0.2"
    S.delay = "j*2*ms"

    M = StateMonitor(G, "v", record=True)

    run(50*ms)

    plt.plot(M.t/ms, M.v[0], label="Neuron 0")
    plt.plot(M.t/ms, M.v[1], label="Neuron 1")
    plt.plot(M.t/ms, M.v[2], label="Neuron 2")
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.legend()
    plt.show()

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    ax1.plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        ax1.plot([0, 1], [i, j], '-k')
    ax1.set_xticks([0, 1], ['Source', 'Target'])
    ax1.set_ylabel('Neuron index')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-1, max(Ns, Nt))
    ax2.plot(S.i, S.j, 'ok')
    ax2.set_xlim(-1, Ns)
    ax2.set_ylim(-1, Nt)
    ax2.set_xlabel('Source neuron index')
    ax2.set_ylabel('Target neuron index')
    plt.show()

def complex_connectivity():
    N = 10
    G = NeuronGroup(N, "v:1")
    S = Synapses(G, G)
    S.connect(condition="i!=j", p=0.2)

    visualise_connectivity(S)


def complex_connectivity_two():
    N = 10
    G = NeuronGroup(N, "v:1")

    for p in [0.1, 0.5, 1.0]:
        S = Synapses(G, G)
        S.connect(condition="i!=j", p=p)
        visualise_connectivity(S)

def complex_connectivity(connection_condition):
    N = 10
    G = NeuronGroup(N, "v:1")

    S = Synapses(G, G)
    S.connect(condition=connection_condition, skip_if_invalid=True)
    visualise_connectivity(S)

def complex_connectivity_j(connection_index):
    N = 10
    G = NeuronGroup(N, "v:1")
    S = Synapses(G, G)
    S.connect(j=connection_index, skip_if_invalid=True)
    visualise_connectivity(S)

def complex_connectivity_spatial():
    N = 30
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing

    G = NeuronGroup(N, "x : metre")
    G.x = "i*neuron_spacing"

    S = Synapses(G, G, "w : 1")
    S.connect(condition="i!=j")
    S.w = "exp(-(x_pre-x_post)**2/(2*width**2))"
    plt.scatter(S.x_pre/um, S.x_post/um, S.w*20)
    plt.xlabel("Source Neuron Position (um)")
    plt.ylabel("Destination Neuron Position (um)")
    plt.show()

def stdp():
    taupre = taupost = 20*ms
    wmax = 0.01
    Apre = 0.01
    Apost = -Apre*taupre/taupost*1.05

    G = NeuronGroup(2, "v:1", threshold="t>(1+i)*10*ms", refractory=100*ms)
    S = Synapses(G, G,
            # Event-driven = we update continouously on spikes, rather than checking if they need updating each time step
            """
                w : 1
                dapre/dt = -apre/taupre : 1 (event-driven)
                dapost/dt = -apost/taupost : 1 (event-driven)
            """,
            # what we should do on a presynaptic spike
            on_pre="""
                v_post += w
                apre += Apre
                w = clip(w+apost, 0, wmax)
            """,
            # what we should do on a postsynaptic spike
            on_post="""
                apost += Apost
                w = clip(w+apre, 0, wmax)
            """,
            method="linear")

    S.connect(i=0, j=1)
    M = StateMonitor(S, ["w", "apre", "apost"], record=True)

    run(30*ms)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(M.t/ms, M.apre[0], label="apre")
    ax1.plot(M.t/ms, M.apost[0], label="apost")
    ax1.set_xlabel("Time (ms)")


    ax2.plot(M.t/ms, M.w[0], label="apost")
    ax2.set_xlabel("Time (ms)")
    plt.show()

def stdp_verification():
    taupre = taupost = 20*ms
    Apre = 0.01
    Apost = -Apre*taupre/taupost*1.05
    tmax = 50*ms
    N = 100

    G = NeuronGroup(N, "tspike:second", threshold="t>tspike", refractory=100*ms)
    H = NeuronGroup(N, "tspike:second", threshold="t>tspike", refractory=100*ms)
    G.tspike = "i*tmax/(N-1)"
    H.tspike = "(N-1-i)*tmax/(N-1)"

    S = Synapses(G, H,
             '''
             w : 1
             dapre/dt = -apre/taupre : 1 (event-driven)
             dapost/dt = -apost/taupost : 1 (event-driven)
             ''',
             on_pre='''
             apre += Apre
             w = w+apost
             ''',
             on_post='''
             apost += Apost
             w = w+apre
             ''')

    S.connect(j="i")
    run(tmax+1*ms)

    plt.plot((H.tspike-G.tspike)/ms, S.w)
    plt.xlabel(r"$\Delta t$ (ms)")
    plt.ylabel(r"$\Delta w$")
    plt.axhline(0, ls="-", c="k")
    plt.show()

if __name__ == "__main__":
    #simple_synapse()
    #adding_weight()
    #introducing_delay()
    #complex_connectivity()
    #complex_connectivity_two()
    #complex_connectivity("abs(i-j)<4 and i!=j")
    #complex_connectivity_j("k for k in range(i-3, i+4) if i!=k")
    #complex_connectivity_j("i")
    #complex_connectivity_spatial()
    #stdp()
    stdp_verification()
