# Seminar 3

## How was the video?
Generally good.
Historical timeline was really good.

SNN vs ANN? What is the difference?


## Seminars

### Brain
How do they estimate the number of neurons and synapses?

### Von Neumann vs. Neuromoprhic
 * Focus: Carver Mead paper "Neuromorphic Electronic Systems".
 * Drawing from first principals in physics and electronics. He argues that we need this for energy optimization.
 * A reasonable figure for energy decrease from NC 10^4
 * Basic Von Neumann architecture
	- Inheriently clock driven
	- Inherinently synchronous
 * Von Neumann bottleneck
	- Seperate processing and memory
 * Neuromorphic Architecture
	- In memory processing
	- We care about timings, rather than numbers
	- Event-based and parallel
	- Analog not digital
	- Generate exponentials from the physics of the processor itself

## Homework
Professor Jaeger.
See videos defined in onedrive
Read paper in onedrive.



### Dynamics System 1
[Link](https://www.youtube.com/watch?v=WZYd75Diduc)

#### What are dynamical system?
Anything that inoloves time
Examples:
	* The universe
	* Life on earth
	* A bitstream
	* Mathematics

Its impossible to not be a dynamical systems

Funamental decisions before modeling starts:
	1. selection: what subsystem is moduled
	2. perceptive: what aspects of the subsystem are modeled

Modeling tools:
	1. Symbolic vs Numerical
	2. Deterministic vs Non-deterministic/Stochastic
	3. Autonomous vs Non-autonomous
	4. Low-dimensional vs High-dimensional
	5. Descrete time vs Contiunous time
	6. Linear vs non-linear
	7. Homogeneous vs non-homogeneous
	8. Stationary, vs non-stationary/evolutionary

#### Basic ingredients
...., x(n-2), x(n-1), x(n), x(n+1), x(n+2) ....., x(n+1), x(n+2) .....
we can have output, u(t), at each timestep, input, u(t), at each timestep and we need something that moves us forward in time, T.
Sequence of states = trajectory

Two views, natural science view: wants a system that is isolated so we can look at the system alone. The engineering view: Important that the system functions correctly in the end (do not care about the internal states). Reliable input -> output system.

#### A zoo of Finite-state models
	1. Deterministic finite-state automa (DFA)
	2. Moore and Maely-machines
	3. Non-deterministic finite-state automata
	4. Finite-dimensional makrov chains
	5. Controlled markov chains
	6. Hidden markov models
	7. Controlled hidden markov model, also known as Partially Observable Markov Decisions processes


### Dynamics System 2
[Link](https://www.youtube.com/watch?v=gQOu6NTDBGo)

#### Cellular Automata (CA)
	* Can be defined in 1-dim, 2-dim..., systems
	* Popular for pattern formation
	* Simpelest kind of model for spatio-temporal dynamics

#### Petri-net
	* Modelling spatially distributed systems with flows and transformations of material or information
	* Components:
		- Places (physical places or agents)
		- Transactions (production thing needs different tokens to finish)
		- Arcs
		- Tokens (things agents can move)
	* Wide diversity
		- Asynch switching
		- Parallel programming
		- Transportation and manufactring logistics
		- Business processes

#### Dynamical Baqyesian networks
Idea: describe the state of stochastic DS at time N by the values of a finite collection of finite-valued input, hidden and output.
Problem: non-termporal. Unfold the "network" to the next timestep instead of creating "circles" in the architecture.

#### A mirage of continouous-state models
There is a difference between continouous time and discrete time.
***General: Modeling becomes easier if you can make the system linear, or make the system appear linear for a shorter amount of time.***

1. [Iterated function systems](https://en.wikipedia.org/wiki/Iterated_function_system)
2. [Ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation)
3. [Stochastic differential equations](https://en.wikipedia.org/wiki/Stochastic_differential_equation)
4. [Delay differntial equations](https://en.wikipedia.org/wiki/Delay_differential_equation)
5. [Field equations](https://en.wikipedia.org/wiki/Field_equation)

#### States
***Classic State Idea***
"A real (idealized, isolated) physical system envolves in continouous time"
At each point in time t, the system is in a state x(t).

Problems:
1. Time itself
2. Ontological status of state
3. Freedom in modeling

***Other State Ideas***
1. Creating state representation by delay embeddings (Takens' Theorem)
	- We can model a high dimensional system from low dimensional observations using a time delay.

### Dynamics System 3
[Link](https://www.youtube.com/watch?v=4s-f7Wzp92U)


