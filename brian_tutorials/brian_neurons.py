from brian2 import *
import matplotlib.pyplot as plt

def a_simple_model():
    tau = 10*ms
    eqs = """
    dv/dt = (1-v)/tau : 1
    """

    G = NeuronGroup(1, eqs, method="exact")
    M = StateMonitor(G, "v", record=True)

    print("Before v = %s" % G.v[0])
    run(100*ms)
    print("After v = %s" % G.v[0])
    print("Expected value of v = %s" % (1-exp(-100*ms/tau)))

    plt.plot(M.t/ms, 1-exp(-M.t/tau), "C1--", label="Analytic")
    plt.plot(M.t/ms, M.v[0], "C0", label="Brian")
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.legend()
    plt.show()

    plt.clf()

def euler_method():
    tau = 10*ms
    eqs = """
        dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
    """

    G = NeuronGroup(1, eqs, method="euler")
    M = StateMonitor(G, "v", record=0)

    G.v = 5

    run(60*ms)

    plt.plot(M.t/ms, M.v[0])
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.show()

    plt.clf()

def spikes_and_refactory_rate():
    tau = 10*ms
    eqs = """
        dv/dt = (1-v)/tau : 1 (unless refractory)
    """

    G = NeuronGroup(1, eqs, threshold="v>0.8", reset="v = 0", refractory=5*ms, method="exact")
    M = StateMonitor(G, "v", record=0)
    spikemon = SpikeMonitor(G)

    run(50*ms)
    print("Spike times: %s" % spikemon.t[:])
    plt.plot(M.t/ms, M.v[0])
    for t in spikemon.t:
        plt.axvline(t/ms, ls="--", c="C1", lw=3)
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.show()

    plt.clf()

def multiple_neurons():
    N = 100
    tau = 10*ms
    eqs = """
        dv/dt = (2-v)/tau : 1
    """

    G = NeuronGroup(N, eqs, threshold="v>1", reset="v=0", method="exact")
    G.v = "rand()"

    spikemon = SpikeMonitor(G)

    run(50*ms)
    plt.plot(spikemon.t/ms, spikemon.i, ".k")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.show()

    plt.clf()

def parameters():
    N = 100
    tau = 10*ms
    v0_max = 3.
    duration = 1000*ms

    eqs = """
        dv/dt = (v0-v)/tau : 1 (unless refractory)
        v0 : 1
    """

    G = NeuronGroup(N, eqs, threshold="v>1", reset="v=0", refractory=5*ms, method="exact")
    M = SpikeMonitor(G)

    G.v0 = "i*v0_max/(N-1)"

    run(duration)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(M.t/ms, M.i, ".k")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Neuron index")

    ax2.plot(G.v0, M.count/duration)
    ax2.set_xlabel("v0")
    ax2.set_ylabel("Firing rate (sp/s)")
    plt.show()

def stochastic_neurons():
    N = 100
    tau = 10*ms
    v0_max = 3.
    duration = 1000*ms
    sigma = 0.2

    eqs = """
        dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
        v0 : 1
    """

    G = NeuronGroup(N, eqs, threshold="v>1", reset="v=0", refractory=5*ms, method="euler")
    M = SpikeMonitor(G)

    G.v0 = "i*v0_max/(N-1)"
    run(duration)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(M.t/ms, M.i, ".k")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Neuron index")

    ax2.plot(G.v0, M.count/duration)
    ax2.set_xlabel("v0")
    ax2.set_ylabel("Firing rate (sp/s)")
    plt.show()


def end_of_tut():
    N = 1000
    tau = 10*ms
    vr = -70*mV
    vt0 = -50*mV
    delta_vt0 = 5*mV
    tau_t = 100*ms
    sigma = 0.5*(vt0-vr)
    v_drive = 2*(vt0-vr)
    duration = 100*ms

    eqs = '''
    dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
    dvt/dt = (vt0-vt)/tau_t : volt
    '''

    reset = '''
    v = vr
    vt += delta_vt0
    '''

    G = NeuronGroup(N, eqs, threshold='v>vt', reset=reset, refractory=5*ms, method='euler')
    spikemon = SpikeMonitor(G)
    M = StateMonitor(G, "v", record=0)

    G.v = 'rand()*(vt0-vr)+vr'
    G.vt = vt0

    run(duration)
    print(len(spikemon.t/ms), len(list(ones(len(spikemon))/(N*defaultclock.dt))))
    plt.plot(M.t/ms, M.v[0])
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.show()

if __name__ == "__main__":
    #a_simple_model()
    #euler_method()
    #spikes_and_refactory_rate()
    #mulitple_neurons()
    #parameters()
    #stochastic_neurons()
    end_of_tut()
