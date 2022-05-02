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

### [Simulation of SNNs with Brian2. Work through the tutorial](https://brian2.readthedocs.io/en/stable/resources/tutorials/index.html)
See: seminar\_6.py


### P vs NP and complexity 10 minute seminar
See: time-complexity\_p\_vs\_np.pdf



## Seminar
