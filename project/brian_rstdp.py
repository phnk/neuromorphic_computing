# Originally from: https://gist.github.com/ferqui/a96a7120c9eb61747987b8b021dce7fc
from brian2 import *

seed(50)

def rasterplot(ax, x, y, x_label, y_label):
# Function used to plot spike times
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
#    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([], [])
    ax.scatter(x, y, marker='|')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Parameters
simulation_duration = 60 * ms

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
gmax = 5.0
dApre = .125
dApost = -.25

## Dopamine signaling
tauc = 256*ms
taud = 1*ms
taus = 1*ms
epsilon_dopa = 1


defaultclock.dt = 0.0625*ms

## Stimuli section
input_indices = array([0, 0, 0, 0, 1, 1,
                       1, 1, 1, 1, 1, 1])
input_times = array([ 1,  6, 16, 26, 7, 17,
                     19, 21, 23, 25, 27, 29])*ms

# Spike generator -> neurons
#input = SpikeGeneratorGroup(2, input_indices, input_times)


input = PoissonGroup(1, rates=100*Hz)

neurons = NeuronGroup(2, '''
                        dv/dt = (ge * (Ee-v) + El - v) / taum : volt
                        dge/dt = -ge / taue : 1
                        ''',
                      threshold='v>vt', reset='v = vr',
                      method='euler')
neurons.v = vr
neurons_monitor = SpikeMonitor(neurons)

synapse = Synapses(input, neurons, 
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )

synapse.connect(i=[0, 0], j=[0, 1])
synapse.w[0] = "0.2 * gmax"
synapse.w[1] = "0.5 * gmax"

@network_operation(dt=10*ms, when="end")
def f(t):
    synapse.w[:][0] += 0.5
    synapse.w[:][1] -= 0.5

# Simulation
mon = StateMonitor(synapse, 'w', record=[0, 1])
input_monitor = SpikeMonitor(input)
outputs_monitor = SpikeMonitor(neurons)

run(simulation_duration, report='text')

print(f"number of input spikes: {len(input_monitor)}")
print(f"number of input spikes: {len(outputs_monitor.spike_trains()[0])}")
print(f"number of input spikes: {len(outputs_monitor.spike_trains()[1])}")

ax1 = plt.subplot(511)
ax1.plot(mon.t/ms, mon.w[0])
ax1.set_ylabel("Synapse 0 weight")

ax2 = plt.subplot(512, sharex=ax1)
ax2.plot(mon.t/ms, mon.w[1])
ax2.set_ylabel("Synapse 1 weight")

ax3 = plt.subplot(513, sharex=ax1)
rasterplot(ax3, input_monitor.t/ms, [1]*len(input_monitor.t/ms), "", "Input spikes")

print(outputs_monitor.spike_trains()[0])
print(outputs_monitor.spike_trains()[1])

ax4 = plt.subplot(514, sharex=ax1)
rasterplot(ax4, outputs_monitor.spike_trains()[0]/ms, [1]*len(outputs_monitor.spike_trains()[0]/ms), "", "Spikes output\nneuron 0")

ax5 = plt.subplot(515, sharex=ax1)
rasterplot(ax5, outputs_monitor.spike_trains()[1]/ms, [1]*len(outputs_monitor.spike_trains()[1]/ms), "", "Spikes output\nneuron 1")


"""
ax = plt.subplot(812)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.d.T/gmax, 'r-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Extracellular\ndopamine d(t)')
ax.set_xticks([])

ax = plt.subplot(813)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.s.T/gmax, 'b-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Eligibility\ntrace c(t)')
ax.set_xticks([])

ax = plt.subplot(814)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.Apre.T/gmax, 'b-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Apre(t)')
ax.set_xticks([])

ax = plt.subplot(815)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.Apost.T/gmax, 'b-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Apost(t)')
ax.set_xticks([])

ax = plt.subplot(816)
ax.plot(synapse_stdp_monitor.t/ms, synapse_stdp_monitor.s.T/gmax, 'g-')
ax.set_xlim([0, simulation_duration/ms])
ax.set_ylabel('Synaptic\nstrength s(t)')
ax.set_xlabel('Time (s)')

#ax = plt.subplot(818)
#rasterplot(ax, M3.t/ms, [1]*len(M3.t/ms))
#ax.set_ylabel("Spikes\noutput neuron")
"""
plt.show()
