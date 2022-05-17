# Seminar 7

## Homework
[Neuromorphic Learning Lecture](https://tube.switch.ch/videos/RaNHxtfdVe)
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


[Neural Coding In the book](https://neuronaldynamics.epfl.ch/online/Ch7.S6.html) 

[SpyTorch: Surrogate gradient learning in spiking neural networks in 20 minutes with PyTorch](https://www.youtube.com/watch?v=xPYiAjceAqU)

[SpyTorch](https://github.com/fzenke/spytorch)

## Seminars
