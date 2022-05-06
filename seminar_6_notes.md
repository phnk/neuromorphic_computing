# Seminar 6

## Homework
### [Neurotech lecture on how we can simulate spiking neural networks](https://tube.switch.ch/videos/XikCG0f44E)

#### [NEST simulator: 25 years of experience](https://www.nest-simulator.org/)
* Farley & Clark (1954) 70 years ago we started looking at spiking neurons
* Hybrid NPI-Thread parallellization
* PyNEST
* Pushing towards exascale calculations
* Using NEST to benchmark. Paper together with SpiNNeker
* Local Cortical Microcircuit
	- What does this mean?
* Standardized software good
	- He states its good because other developers can pick it up easier
	- But a really good reason to use standarized software is that we know that there is no implementation differences that are relevent for performance increases
* NEST is reaching faster than biological time
* Lower energy consumption even if we got faster simulation time
* Can create new models using their domain specific language, NESTML, rather than writing c++ code
* Has a GUI which has been used for teaching
* [PyNN](https://neuralensemble.org/PyNN/) is a universal gateway that can run on many different backends (NEST, Brian, etc)


#### [The Brian simulator](https://brian2.readthedocs.io/en/stable/)
* "A simulator should not only save the time of processors, but also the time of scientists"
* Tool to describe "any" model, rather than having predefined models
* Use equations to describe the models. Same as in scinetific publications
* Use physical units
* Python
* Python = really slow. Solution: Use python to describe things -> generate code into a c++ or c -> run simulation
* Runtime mode
	1. Main loop still in python
		- Positive: Highly flexible
		- Negative: High overhead
* Standalone mode
	1. Everything is written to C++
		- Positive: Fast, no python overhead
		- Positive: Can be tailored to other platforms
		- Negative: Less flexible, no python interaction during run
* Brian's domain
	- Laptops (small models)
	- Can simulation LIF models and multi-compartemental models (unless complex morphologies)
	- No good parallalization hence bad for big computer clusters and to simulation very large models


#### [GeNN](https://genn-team.github.io/)
* GPU enhanced neuronal networks using CUDA
* Mapping SNNs to GPUs is non-trivial, especially when we want to do this for effiency
* Large ecosystem where there are many interfaces such as Brian2GeNN.
* How does it work?
	- We have neurons population and synapse populations
* Example in the video
* Why GeNN?
	- If you want a faster generation for GPU
	- Very flexibile
	- Allows for custom neurons, synapses, learning, etc
	- Full control of the simulation loop
	- Crossplatform
* When GeNN?
	- Large spiking models
	- Long simulation time
	- "Realtime" on robots

### [IAIFI Colloquium Series: A path towards human-level intelligence](https://www.youtube.com/watch?v=_K0h7oEe8BQ)
* How can we teach machines to learn like humans and animals as they are way more efficient
* Three challenges in AI
	- Learning representation and predictive models of the world
	- Learning to reason
	- Learning to plan complex action sequences
		- Hierarchical planning
* How do animals and humans learn quickly?
	- No supervision, no reinforcement
	- Babies: Observe the world, learn how it works
	- Passive learning
	- Background knowledge
	- How do we reproduce this process in machines?
* Machines need to get into reality to learn "common sense"
* Common sense is a collection of models of the world
	- Allows us to reason
	- "The nature of Explanation" by Kenneth Craik
* Yanns Cake
	- RL
	- Supervised learning
	- Self supervised learning (bulk of the biological learning)
* Modular architecture for Autonomous AI
	- Configurator: configures other modules for tasks
	- Perception: Estiamtes state of the world
	- World model: Predicts future world states
	- Cost: Compute "discomfort". Two costs, intrinsic cost and critic cost.
	- Actor: Find optimal action sequences
	- Short-term memory: Stores state-cost episodes
* Different ways to use this model
	- Node-1: Perception-Action cycle
	- Node-2: Perception-Planning-Action cycle
* How is the cost module built?
	- Instrinsic cost: Immutable. Hard-wired.
	- Trainable cost: Trainable. Predicts future values of the instrinsic cost from the current state.
	- Some linear combination?
* Train the critic by some short term memory to preduce training pairs.
* Main question: How do we build and train the world model
	- The world is stochastic (or deterministic but not predictable)
	- How dowe represent uncertainty in the predictions?
* Self-supervised learning
	- Fill in the blanks in the observation in time and/or space
	- Learning hierarchical representations of the world
	- Learning predictive (forward) models of the world
* Energy-based models: Implicit function
	- Capture dependencies through an energy function
	- A way to not deal with untractable partition function
	- Energy function, based on neural network, that gives low energy for compatiable paris of x and why, and a high energy for incompatiable pairs.
	- Multiple predictions per input
	- High energy for things we have not observed, low energy for things we have observed
* Energy-based models: Inference by Minimization
	- Find the value of y that makes F(x, y) small
	- Energy is used for inference, not for learning
* Conditional and Unconditional Energy-based models
	- Conditional: F(x, y)
	- Unconditional: F(y)
* Probalistic models are a special case of Energy-based models
	- Physics/math talk I do not understand
* Latent-variable EBM: z
	- Minimize over it so it is not used in the energy function
* Example of a unconditional latent-variable EBM: K-Means
* How do we train energy-based models?
	- Need to avoid a collapse
	- Good visualization that explains it in the video (32:45-ish)
	- Collapse, Contrastive Method, Regularized Methods
* EBM architectures can collapse or not depending on architecture. However non collapsable architectures is not very interesting.
	- Joint embedding architecture seems the most interesting.
* How do we prevent our EBM to collapse during training?
	- Contrastive methods: Push down on training samples, up on non-traning samples. Scales poorly with dimensions. Doomed according to Yann
	- Regularized methods: Regularizer minimizes the volume of space that can take low energy
* Contrastive methods: Different ways to pick which points to push up
	- Denoising AE, GANs, Masked AE etc
* Regularized: Different ways to control the information capacity of the latent representation
	- PCA, K-means, Sparse coding, sparse AE, score matching, VAE
	- Focus: VICReg: TODO: reference
* Many loss functions for contrastive methods and you do not want to use them
* Problem with Max likelihood / Probalistic Methods
	- Wants to make the difference between the energy on the data manifold and the energy just outside of it infinitely large!
	- Think deep canyon
	- Can't use for inference because it does not have a good gradient
* What architectures can we use to build energy based models?
	- JEA
	- DJEPA
	- JEPA
* Herarchical JEPA



### [Simulation of SNNs with Brian2. Work through the tutorial](https://brian2.readthedocs.io/en/stable/resources/tutorials/index.html)
See: seminar\_6.py


### P vs NP and complexity 10 minute seminar
See: time-complexity\_p\_vs\_np.pdf

## Seminar

### Seminar (Hannah)
* Simulators (SNN)
* Paper goal: Guidance for users to choose SNN simulators for development and guidance in development of future SNNs simulators.
* Tested 5 simulators on non-computational neuroscience tasks. Tested of speed, scalability and flexability.
	- NEST, Brian2, Brian2GeNN, NengoLoihi, BindsNET
	- Check BindsNET as the focus is on ML/RL? (How does this work?, what do we mean?)
	- All different simulators has different adventages and disadvantages

### Seminar (Mine)
No comments

### Seminar (Akshit)
* Event-based Vision
* Inspired by the human vision
	- Cones (Color) and rods (Night vision)
	- Only reports the change in the scene, not the scene itself
	- Efficient way to decrease the data transfer
	- Riemann (discretise in the x axis) vs Lebesgue (discretise in the y axis) Q: How is this used in real systems? For me its easy to internalise riemann sampling but not the other
* Weakness of conventional vision
	- Moving objects undersampled, background oversampled
	- Overwhelmed with irrelevent data
* Strength of Neuromorphic Vision
	- "Volume of data goverened by the activity in the field of view"
	- Good for SLAM. Q: has someone done this yet?
* Implications for Robotics and ML
	- Feedback control loop can run at high frequenices
