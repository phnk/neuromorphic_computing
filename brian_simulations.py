# from: https://brian2.readthedocs.io/en/stable/resources/tutorials/3-intro-to-brian-simulations.html

from brian2 import *
import matplotlib.pyplot as plt

def multiple_runs():
    num_inputs = 100
    input_rate = 10*Hz
    weight = 0.1
    tau_range = linspace(1, 10, 30)*ms
    output_rates = []

    P = PoissonGroup(num_inputs, rates=input_rate)
    eqs = """
        dv/dt = -v/tau : 1
    """
    G = NeuronGroup(1, eqs, threshold="v>1", reset="v=0", method="exact")
    S = Synapses(P, G, on_pre="v += weight")
    S.connect()

    M = SpikeMonitor(G)

    store()

    for tau in tau_range:
        restore()
        run(1*second)

        output_rates.append(M.num_spikes/second)

    plt.plot(tau_range/ms, output_rates)
    plt.xlabel(r"$\tau$ (ms)")
    plt.ylabel("Firing rate (sp/s)")
    plt.show()

def multiple_runs_store_spikes():
    start_scope()
    num_inputs = 100
    input_rate = 10*Hz
    weight = 0.1
    tau_range = linspace(1, 10, 30)*ms
    output_rates = []
    # Construct the Poisson spikes just once
    P = PoissonGroup(num_inputs, rates=input_rate)
    MP = SpikeMonitor(P)
    # We use a Network object because later on we don't
    # want to include these objects
    net = Network(P, MP)
    net.run(1*second)
    # And keep a copy of those spikes
    spikes_i = MP.i
    spikes_t = MP.t
    # Now construct the network that we run each time
    # SpikeGeneratorGroup gets the spikes that we created before
    SGG = SpikeGeneratorGroup(num_inputs, spikes_i, spikes_t)
    eqs = '''
    dv/dt = -v/tau : 1
    '''
    G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    S = Synapses(SGG, G, on_pre='v += weight')
    S.connect()
    M = SpikeMonitor(G)
    # Store the current state of the network
    net = Network(SGG, G, S, M)
    net.store()
    for tau in tau_range:
        # Restore the original state of the network
        net.restore()
        # Run it with the new value of tau
        net.run(1*second)
        output_rates.append(M.num_spikes/second)
    plt.plot(tau_range/ms, output_rates)
    plt.xlabel(r'$\tau$ (ms)')
    plt.ylabel('Firing rate (sp/s)');
    plt.show()

def multiple_runs_no_loop():
    start_scope()
    num_inputs = 100
    input_rate = 10*Hz
    weight = 0.1
    tau_range = linspace(1, 10, 30)*ms
    # save the length of tau
    num_tau = len(tau_range)
    P = PoissonGroup(num_inputs, rates=input_rate)
    # We make tau a parameter of the group
    eqs = '''
    dv/dt = -v/tau : 1
    tau : second
    '''
    # And we have num_tau output neurons, each with a different tau
    G = NeuronGroup(num_tau, eqs, threshold='v>1', reset='v=0', method='exact')
    # set G.tau to the tau range we saved earlier to allow us to not have to use a loop during the simulation
    G.tau = tau_range
    S = Synapses(P, G, on_pre='v += weight')
    S.connect()
    M = SpikeMonitor(G)
    # Now we can just run once with no loop
    run(1*second)
    output_rates = M.count/second # firing rate is count/duration
    plt.plot(tau_range/ms, output_rates)
    plt.xlabel(r'$\tau$ (ms)')
    plt.ylabel('Firing rate (sp/s)');
    plt.show()

    plt.clf()
    trains = M.spike_trains()
    isi_mu = full(num_tau, nan)*second
    isi_std = full(num_tau, nan)*second
    for idx in range(num_tau):
        train = diff(trains[idx])
        if len(train)>1:
            isi_mu[idx] = mean(train)
            isi_std[idx] = std(train)
    plt.errorbar(tau_range/ms, isi_mu/ms, yerr=isi_std/ms)
    plt.xlabel(r'$\tau$ (ms)')
    plt.ylabel('Interspike interval (ms)')
    plt.show()

def current_injection_during_run():
    start_scope()
    # Parameters
    area = 20000*umetre**2
    Cm = 1*ufarad*cm**-2 * area
    gl = 5e-5*siemens*cm**-2 * area
    El = -65*mV
    EK = -90*mV
    ENa = 50*mV
    g_na = 100*msiemens*cm**-2 * area
    g_kd = 30*msiemens*cm**-2 * area
    VT = -63*mV
    # The model
    eqs_HH = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
        (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
        (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
        (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    I : amp
    '''
    group = NeuronGroup(1, eqs_HH,
                        threshold='v > -40*mV',
                        refractory='v > -40*mV',
                        method='exponential_euler')
    group.v = El
    statemon = StateMonitor(group, 'v', record=True)
    spikemon = SpikeMonitor(group, variables='v')
    # we use this network_operation (instead of a loop) to inject external current into our single neuron
    @network_operation
    def change_I():
        group.I = rand()*50*nA

    run(50*ms)
    plt.figure(figsize=(9, 4))
    # loop only left for the plotting
    for l in range(5):
        plt.axvline(l*10, ls='--', c='k')
    plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
    plt.plot(statemon.t/ms, statemon.v[0]/mV, '-b')
    plt.plot(spikemon.t/ms, spikemon.v/mV, 'ob')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)');
    plt.show()

def current_injection_multiple_neurons():
    start_scope()
    N = 3
    area = 20000*umetre**2
    Cm = 1*ufarad*cm**-2 * area
    gl = 5e-5*siemens*cm**-2 * area
    El = -65*mV
    EK = -90*mV
    ENa = 50*mV
    g_na = 100*msiemens*cm**-2 * area
    g_kd = 30*msiemens*cm**-2 * area
    VT = -63*mV
    eqs_HH_2 = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/C : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
        (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
        (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
        (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    I : amp (shared) # added this to FORCE the neurons I to be shared, hence we inject external I at the same time for each and every neuron
    C : farad
    '''
    group = NeuronGroup(N, eqs_HH_2,
                        threshold='v > -40*mV',
                        refractory='v > -40*mV',
                        method='exponential_euler')
    group.v = El
    # initialise with some different capacitances
    group.C = array([0.8, 1, 1.2])*ufarad*cm**-2*area
    statemon = StateMonitor(group, variables=True, record=True)
    # we go back to run_regularly
    group.run_regularly('I = rand()*50*nA', dt=10*ms)
    run(50*ms)
    plt.figure(figsize=(9, 4))
    for l in range(5):
        plt.axvline(l*10, ls='--', c='k')
    plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
    plt.plot(statemon.t/ms, statemon.v.T/mV, '-')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.show()

    plt.clf()
    plt.plot(statemon.t/ms, statemon.I.T/nA, "-")
    plt.xlabel('Time (ms)')
    plt.ylabel('I (nA)')
    plt.show()

def adding_input():
    start_scope()
    A = 2.5
    f = 10*Hz
    tau = 5*ms
    # create a timedarray and set the equations to use it (can be used to load input data from file)
    t_recorded = arange(int(200*ms/defaultclock.dt))*defaultclock.dt
    I_recorded = TimedArray(A*sin(2*pi*f*t_recorded), dt=defaultclock.dt)
    eqs = '''
    dv/dt = (I-v)/tau : 1
    I = I_recorded(t) : 1
    '''
    G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    M = StateMonitor(G, variables=True, record=True)
    run(200*ms)
    plt.plot(M.t/ms, M.v[0], label='v')
    plt.plot(M.t/ms, M.I[0], label='I')
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.legend(loc='best')
    plt.show()

def weird_input():
    start_scope()
    A = 2.5
    f = 10*Hz
    tau = 5*ms
    # Let's create an array that couldn't be
    # reproduced with a formula
    num_samples = int(200*ms/defaultclock.dt)
    I_arr = zeros(num_samples)
    for _ in range(100):
        a = randint(num_samples)
        I_arr[a:a+100] = rand()
    I_recorded = TimedArray(A*I_arr, dt=defaultclock.dt)
    eqs = '''
    dv/dt = (I-v)/tau : 1
    I = I_recorded(t) : 1
    '''
    G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    M = StateMonitor(G, variables=True, record=True)
    run(200*ms)
    plt.plot(M.t/ms, M.v[0], label='v')
    plt.plot(M.t/ms, M.I[0], label='I')
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.legend(loc='best')
    plt.show()

def reading_from_file():
    start_scope()
    from matplotlib.image import imread
    img = (1-imread('brian-logo.png'))[::-1, :, 0].T
    num_samples, N = img.shape
    ta = TimedArray(img, dt=1*ms) # 228
    A = 1.5
    tau = 2*ms
    eqs = '''
    dv/dt = (A*ta(t, i)-v)/tau+0.8*xi*tau**-0.5 : 1
    '''
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='euler')
    M = SpikeMonitor(G)
    run(num_samples*ms)
    plt.plot(M.t/ms, M.i, '.k', ms=3)
    plt.xlim(0, num_samples)
    plt.ylim(0, N)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()

if __name__ == "__main__":
    #multiple_runs()
    #multiple_runs_store_spikes()
    #multiple_runs_no_loop()
    #current_injection_during_run()
    #current_injection_multiple_neurons()
    #adding_input()
    #weird_input()
    reading_from_file()
