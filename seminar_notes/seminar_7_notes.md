# Seminar 7

# Homework
## [Neuromorphic Learning Lecture](https://tube.switch.ch/videos/RaNHxtfdVe)
### Online Learning
* Offline learning: We decide some timestep where we collect all experiance and then from that decide how to update our parameters
* Online learning: We want to update the parameters after each relevant time step (more suitable for NC)
* Hopfield model as example
* Problems with online learning in NC
	- Memory capacity becomes a large problem (scaling)
	- Memory scales linearly with the parameters (lower case)
	- However we need very precise synapses
	- Limited precision makes memory scale log with the parameters which we do not want. We want memory to be able to contain a high amount of parameters
	- Large number of synapses but can't take advantage of it (due to memory)
	- But we can change the learning rate (low learning rate) for slow learning.
	- We want a nice scaling for our signal to noise ratio AND nice scaling for the number of memories.
	- Simple solution
		1. Divide our population into sub population
		2. Write to them at the same time
		3. Different learning rate
		4. We can write very rapitly and we can preserve the memory at a longer timescale (we need multiple time scales in our system)
	- Still slow as we are writing to all at the same time
	- Different strategy
		1. Write to only the first (fast) population
		2. Cascade the information to the other population.
		3. p: M * sqrt(N), SNR: sqrt(N)
	- We can do better
		1. Write on only the first (fast) one
		2. Bidirectional between all population.
		3. p: N, SNR: sqrt(N) (we can recover the optimal storage capacity here)

### Three tiers of neuromorphic learning
* Motivation: If we work on NC hardware we can learn new things about the algorithms that work well on these types of hardware. Also energy efficiency.
* Many NC platforms (Lohi, Spinnaker, Dynapse, etc)
* 2 groups: Digital systems and Mixed-signal systems
* Say we have designed our network, connectivity, update equations etc. How do we take this to the hardware?
	- Ideal world: everything works (but it doesn't in reality)
	- Tier 1a: Offline learning and then translate the network to "fit" on the NC hardware.
	- Tier 2: In-the-loop training. Works well on analog chips. But we have to go back and forth between a host pc and our chip, hence slowing down training.
	- Tier 3: On-chip learning. The best, but is difficult.
	- Start on Tier 2. Go to Tier 3 with approximations as sometimes we can't use the exact model we have designed. Example used in video is exponential functions.
* Time-to-first-spike coding
	- Error back propagation
	- Requires derivatives, LIF neurons.
	- Too difficult to bring the derivatives to chip for on-chip learning
	- Power consupmtion: 200 mW
	- Classification speed: 10k images/s
	- Slow training however as we need to move back and forth between the host and the chip
	- Solution: Simplify the derivatives and see how much accuracy you lose
	- Works suprisingly well, but this is where they currently are at

### How to use online neuromorphic learning in practice
* Online learning by gradient descent is hard
	- Data is not ideal -> catastrophic forgetting
	- Data inefficiency
	- Real-world is a batch size 1
	- Might be that the tabual rasa approach of deep learning is not made for neuromorphic hardware
* Possible solutions:
	- Online: complex synapses
	- Online: Pseudo-replay
	- Offline: Meta-learning / Transfer learning
	- Offline: Federated learning
* LIF neuron example
	- Basically a RNN if we graph the state changes
	- All learning rules from RNNs can thus be used for LIFs
* Challenges in gradient-based SNNs
	- Non-differentability. Replace the function with a smooth one for approximation. This is: ***Surrogate gradient descent***
	- Deep spatial credit assignment. Solution: Construct approximate learning channels. Other solution: Local learning rules for each layer.
	- Requires temporal credit assignment. Approaches: BPTT and Real-time recurrent learning. 
	- BPTT: Offline and non-local but works (most focus on BPTT).
	- Real-time recurrent learning: Online and non-local but very memory intensive.
	- Can mix these two. Scales better, online and local  (Deep Continuous On Line Learning (DECOLLE).
* Done by pretrain on GPU > transfer to Loihi chip > can learn new gestures after only a single shot
* However, because its an approximation, we CAN create situations where it the approximation does not work
* Meta-training SNNs
* Summary
* Challenges
	- Continual learning is difficult
	- On-chip gradient-based learning is not compatible with the tabula rasa approach
* One approach
	- Gather "dataset of datasets"
	- Pretrain using chip-in-the-loop
	- Perform few-shot learning on hardware
* Advantages of said approach
	- Data efficiency at the edge
	- Compensation for algorithmic approximations and limited precision

### Sparse resorvoir computing
* Restrict learning in one single layer: reservoir computing
	- Input -> random sparse matrix W -> output
	- Define network by a differential equation where we want to capture how the network changes over time
	- Learning only happens in the output weights
	- Only need some function that takes the input and the relevant parameters
* Desirable properties for learning rules
	- We want a local learning rule
	- We want simplicity
	- We want the learning rule to be online
	- We don't want the learning rule to interfere with the reservoir dynamics
* Read-out weights update rule
	 - Online setting and more biologically plausible
	- Threshold learning "kills" correlated neurons
	- We can define the initial sparsity level (but we need to know the distribution of our input)
* Summary
	- Reservior computing offers a useful framework for solving single layer learning problems
	- Reservior computing has benefits when we want to harness the material properties
	- Learning rules should not disrupt the dynamics
	- SPARCE is an example of this


## [Neural Coding In the book](https://neuronaldynamics.epfl.ch/online/Ch7.S6.html) 
* The problem of Neural Coding
	- In the book we use different ways to quantify spike trains (firing rate often used).
	- However we are only measuring the frequency in some way and the question is if neurons transmit other information using the spikes.
	- This type of information is neural code.
* Limitations of spike count coding
	- A fly can react to stimuli in the range of 30-40 ms, not enough time to perform some average over spikes
	- Humans can detect changes in visual stimuli in a few hundred millisteconds. Not enough time to perform any averages.
	- Spikes are just convinient to transmit analog output
	- Temporal averaging only works for constant or slowly varying stimuli
* Limitations of Peri-Stimulus-Time Histogram (PSTH)
	- Relies on multiple trials to build up.
	- Example: A frog that wants to catch a fly cannot wait for the fly to take the same trajectory multiple times to learn how to.
	- Relies on the assumption that there are always populations of neurons with similar properties
* Limitations of rate as a population average
	- Assumption: Homogeneous neuron populations with identical connections. This is hardly realisitic
	- However: rate of population activity may be feasable
	- Potential problem: Neuronal coding and decoding schemes lies in the normalization by division
* Candidate temporal codes
	- Time-to-first-spike: Latency code, where different stimuli yield different spike letencies, yet they are consistent over trials
		1. We only care about the time to first spike. If more spikes come we ignore them
		2. Since we only transmit 1 spike per stimuli it is clear that only the timing is important and not the number of spikes
	- Phase: Same as above, but instead of a single event we see it as a periodic signal where the oscillations serves as an internal reference signal. Then spikes could encode information using the oscillations as a reference.
	- Correlations and Synchrony
		1. We can use spikes from other neurons as reference signal for a spike coding.
		2. Pair of many neurons spiking together can signify a special event
		3. Or we could use pulses where after one neuron spikes, others spike after some small timestep. 
	- Stimulus Reconstruction and Reverse Correlation
		1. We average over some time window (such as 100ms) where the spike is in the middle of said time windiw. We can then create some average used to transport information.
	- Rate versus temporal codes. The divide between spike codes and firing rate is not clearly drawn as some codes can be interpreted or transfored from one into the other.
	- "What is important, in our opinion, is to have a coding scheme which allows neurons to quickly respond to stimulus changes"

## [SpyTorch: Surrogate gradient learning in spiking neural networks in 20 minutes with PyTorch](https://www.youtube.com/watch?v=xPYiAjceAqU)
* We want to build a task-optimized SNN like we do with ANNs
	- Requirements: Input and output
	- After input and output defined, we need to worry about one thing: connectivity. We will decide the connectivity with surrogate gradients
	- Typical input: Spatiotemporal spike trains
	- Typcial output: Linear combination of filtered output spike trains. Look for the maximum of the resulting curves to reduce the temporal compnent
* How do we train?
	- Well as we have learned before, SNNs are RNNs in a computer science sense
	- ***Problem: Derivatives of spikes is ill-defined***, "hard" threshold. Its zero or inifinite.
	- This leads to that there is no gradient, hence no where to walk down in the space. This has the consequence of gradient based methods not working.
	- ***Solution: Surrogate gradients***
	- Whenever we compute we compute derivatives, we compute a surrogate derivative that is smooth
	- This gives us a space that is smooth, hence gradient methods work.
* Notebooks showcase: [SpyTorch](https://github.com/fzenke/spytorch)

# Seminars

## Event-triggered sensor pose estimation
* Pose = the orientation and position. Consider x, y, z. Position is change along some axis. Orientation is the rotation around some axel.
* Even based vision sensor
	- Sparse output consiting of stream of events rather than frames
	- Onboard clock can be used to sample, similar to a normal camera?
	- Only measures change. "pixels" that do not change gives no output
* Espee Algorithm
	- Used to perform pose estimation from event information to some map
* Main goal: use event based cameras by minmizing the error between the event based and lidar to get some error. after we have this error we use EKF to minimize it.

## Summary of Capocaccia workshop
* Started by Caever meed
* Organized: Institute of neuroimformatics, Zurich
* Norse simulator - Check github
* Spike-time-based gradient descent


