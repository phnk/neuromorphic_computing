# Originally from: https://gist.github.com/ferqui/a96a7120c9eb61747987b8b021dce7fc
from brian2 import *
import numpy as np


def rasterplot(ax, x, y, x_label, y_label):
# Function used to plot spike times
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
#    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([], [])
    ax.scatter(x, y, marker='|')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Parameters
simulation_duration = 60*ms

## Neurons
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms

## STDP
taupre = 16*ms
taupost = taupre
gmax = 1.0
dApre = .125
dApost = -.25

## Dopamine signaling
tauc = 256*ms
taud = 1*ms
taus = 1*ms
epsilon_dopa = 1


defaultclock.dt = 0.0625*ms

## Stimuli section

### TODO: come up with a good scenario for the input.
input_indices = array([0, 0, 0, 0, 1, 1,
                       1, 1, 1, 1, 1, 1])
input_times = array([ 1,  6, 16, 26, 7, 17,
                     19, 21, 23, 25, 27, 29])*ms
input = SpikeGeneratorGroup(2, input_indices, input_times)

# to generate input for now
#input = PoissonGroup(2, rates=100*Hz)

### Neurons to decide if we are moving left or right depending on their instant output spike frequency
neurons = NeuronGroup(2, '''
                        dv/dt = (El - v) / taum : volt
                        dge/dt = -ge / taue : 1
                        ''',
                      threshold='v>vt', reset='v = vr',
                      method='exact')
neurons.v = vr


neurons_monitor = SpikeMonitor(neurons)

### Synapse connecting input and output
synapse = Synapses(input, neurons, 
            model="""s: volt""",
            on_pre="v += s"
            )


### Synapse connections: input 0 -> neuron 0, 1. input 1 -> neuron 0, 1. (fully connected, only doing it to get it in a w array order I understand
synapse.connect(i=[0, 1], j=[0, 1])
synapse.s = 100. * mV

#synapse.s[0] = "0.2 * gmax"
#synapse.s[1] = "0.2 * gmax"
#synapse.s[2] = "0.2 * gmax"
#synapse.s[3] = "0.2 * gmax"

### R-STDP synapse connection. In order to not alter the pre and post-synaptic spikes, the weight of this synapse do not change affect the dynamics of the neuron as described in: https://gist.github.com/ferqui/a96a7120c9eb61747987b8b021dce7fc

## STDP section

# c = eligibility trace
# d = dopamine
# s = weight
synapse_stdp = Synapses(neurons, neurons,
                   model='''
                         mode: 1
                         dc/dt = -c / tauc : 1 (clock-driven)
                         dd/dt = -d / taud : 1 (clock-driven)
                         ds/dt = c * d / taus : 1 (clock-driven)
                         dApre/dt = -Apre / taupre : 1 (clock-driven)
                         dApost/dt = -Apost / taupost : 1 (clock-driven)''',
                   on_pre='''ge += s
                          Apre += dApre
                          c = clip(c + Apost, -gmax, gmax)
                          ''',
                   on_post='''Apost += dApost
                          c = clip(c + Apre, -gmax, gmax)
                          ''',
                   method='euler'
                   )

synapse_stdp.connect(i=0, j=1)
synapse_stdp.mode = 0
synapse_stdp.s = 0.9
synapse_stdp.c = 0
synapse_stdp.d = 0

synapse_stdp_monitor = StateMonitor(synapse_stdp, ['s', 'c', 'd', 'Apre', 'Apost'], record=[0])

## Dopamine signaling
# TODO: match the dopamine times with the input spikes
dopamine_indices = array([0])
dopamine_times = array([10])*ms
dopamine = SpikeGeneratorGroup(1, dopamine_indices, dopamine_times)
dopamine_monitor = SpikeMonitor(dopamine)
reward = Synapses(dopamine, synapse_stdp, model='''
                            active : 1
                            ''',
                            on_pre='''d_post += epsilon_dopa * active
                            ''',
                            method='exact')

reward.connect()

reward.active = 0

input_monitor = SpikeMonitor(input)

@network_operation(dt=1*ms)
def reward_function(t):
    # "instant" freq
    freq_0, freq_1 = 0, 0
    dt = 0.0625*ms

    if len(input_monitor.spike_trains()[0]) > 1:
        freq_0 = (input_monitor.spike_trains()[0][-1] - input_monitor.spike_trains()[0][-2])/dt

    if len(input_monitor.spike_trains()[1]) > 1:
        freq_1 = (input_monitor.spike_trains()[1][-1] - input_monitor.spike_trains()[1][-2])/dt

    if freq_0 > freq_1:
        reward.active_ = 1
    elif freq_1 > freq_0:
        reward.active_ = 0
    else:
        pass


run(simulation_duration, report='text')


# Simulation
#mon = StateMonitor(synapse, 'w', record=[0, 1, 2, 3])
#outputs_monitor = SpikeMonitor(neurons)


#print(f"number of input spikes: {len(input_monitor)}")
#print(f"number of input spikes: {len(outputs_monitor.spike_trains()[0])}")
#print(f"number of input spikes: {len(outputs_monitor.spike_trains()[1])}")

# Visualisation
dopamine_indices, dopamine_times = dopamine_monitor.it
neurons_indices, neurons_times = neurons_monitor.it
ax = plt.subplot(611)
#plot([0.05, 2.95], [2.7, 2.7], linewidth=5, color='k')
#text(1.5, 3, 'Classical STDP', horizontalalignment='center', fontsize=20)
#plot([3.05, 5.95], [2.7, 2.7], linewidth=5, color='k')
#text(4.5, 3, 'Dopamine modulated STDP', horizontalalignment='center', fontsize=20)
ax.plot([0, simulation_duration/ms], [0,0], linewidth=1, color='0.5')
ax.plot([0, simulation_duration/ms], [1,1], linewidth=1, color='0.5')
ax.plot([0, simulation_duration/ms], [2,2], linewidth=1, color='0.5')
ax.plot(neurons_times/ms, neurons_indices, 'ob')
ax.plot(dopamine_times/ms, dopamine_indices + 2, 'or')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylim([-0.5, 2.5])
ax.set_yticks([0, 1, 2], ['Pre-neuron', 'Post-neuron', 'Reward'])
ax.set_xticks([])

ax = plt.subplot(612)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.d.T/gmax, 'r-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Extracellular\ndopamine d(t)')
ax.set_xticks([])

ax = plt.subplot(613)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.c.T/gmax, 'b-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Eligibility\ntrace c(t)')
ax.set_xticks([])

ax = plt.subplot(614)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.Apre.T/gmax, 'b-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Apre(t)')
ax.set_xticks([])

ax = plt.subplot(615)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.Apost.T/gmax, 'b-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Apost(t)')
ax.set_xticks([])

ax = plt.subplot(616)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.s.T/gmax, 'g-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Synaptic\nstrength s(t)')
ax.set_xlabel('Time (s)')
plt.show()
#savefig('RSTDP.png')


"""
ax1 = plt.subplot(711)
#ax1.plot(mon.t/ms, mon.w[0])
#ax1.set_ylabel("Synapse 0 weight")

ax = plt.subplot(712, sharex=ax1)
#ax.plot(mon.t/ms, mon.w[1])
#ax.set_ylabel("Synapse 1 weight")

ax = plt.subplot(713)
#ax.plot(mon.t/ms, mon.w[2])
#ax.set_ylabel("Synapse 2 weight")

ax = plt.subplot(714, sharex=ax1)
#ax.plot(mon.t/ms, mon.w[3])
#ax.set_ylabel("Synapse 3 weight")

ax = plt.subplot(715, sharex=ax1)
rasterplot(ax, input_monitor.spike_trains()[0]/ms, [1]*len(input_monitor.spike_trains()[0]/ms), "", "Input spikes")

rasterplot(ax, input_monitor.spike_trains()[1]/ms, [1]*len(input_monitor.spike_trains()[1]/ms), "", "Input spikes")

print(outputs_monitor.spike_trains()[0])
print(outputs_monitor.spike_trains()[1])

ax = plt.subplot(716, sharex=ax1)
rasterplot(ax, outputs_monitor.spike_trains()[0]/ms, [1]*len(outputs_monitor.spike_trains()[0]/ms), "", "Spikes output\nneuron 0")

ax = plt.subplot(717, sharex=ax1)
rasterplot(ax, outputs_monitor.spike_trains()[1]/ms, [1]*len(outputs_monitor.spike_trains()[1]/ms), "", "Spikes output\nneuron 1")
"""

plt.show()
