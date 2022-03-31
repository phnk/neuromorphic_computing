# Seminar 3

## How was the video?
Generally good.
A historical timeline was really good.

SNN vs ANN? What is the difference?


## Seminars

### Brain
How do they estimate the number of neurons and synapses?

### Von Neumann vs. Neuromoprhic
 * Focus: Carver Mead paper "Neuromorphic Electronic Systems".
 * Drawing from first principles in physics and electronics. He argues that we need this for energy optimization.
 * A reasonable figure for energy decrease from NC 10^4
 * Basic Von Neumann architecture
	- Inherently clock-driven
	- Inherently synchronous
 * Von Neumann bottleneck
	- Separate processing and memory
 * Neuromorphic Architecture
	- In-memory processing
	- We care about timings, rather than numbers
	- Event-based and parallel
	- Analog not digital
	- Generate exponentials from the physics of the processor itself

## Homework
Professor Jaeger.
See videos defined in onedrive
Read the paper in onedrive.



### Dynamics System 1
[Link](https://www.youtube.com/watch?v=WZYd75Diduc)

#### What are dynamical systems?
Anything that involves time
Examples:
	* The universe
	* Life on earth
	* A bitstream
	* Mathematics

It is impossible to not be a dynamical system

Fundamental decisions before modelling starts:
	1. selection: what subsystem is moduled
	2. perceptive: what aspects of the subsystem are modelled

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

#### Dynamical Bayesian networks
Idea: describe the state of stochastic DS at time N by the values of a finite collection of finite-valued input, hidden and output.
Problem: non-temporal. Unfold the "network" to the next timestep instead of creating "circles" in the architecture.

#### A mirage of continouous-state models
There is a difference between continuous-time and discrete time.
***General: Modeling becomes easier if you can make the system linear, or make the system appear linear for a shorter amount of time.***

1. [Iterated function systems](https://en.wikipedia.org/wiki/Iterated_function_system)
2. [Ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation)
3. [Stochastic differential equations](https://en.wikipedia.org/wiki/Stochastic_differential_equation)
4. [Delay differntial equations](https://en.wikipedia.org/wiki/Delay_differential_equation)
5. [Field equations](https://en.wikipedia.org/wiki/Field_equation)

#### States
***Classic State Idea***
"A real (idealized, isolated) physical system evolves in continuous time"
At each point in time t, the system is in a state x(t).

Problems:
1. Time itself
2. Ontological status of state
3. Freedom in modelling

***Other State Ideas***
1. Creating state representation by delay embeddings (Takens' Theorem)
	- We can model a high dimensional system from low dimensional observations using a time delay.

### Dynamics System 3
[Link](https://www.youtube.com/watch?v=4s-f7Wzp92U)

States can be defined by what has happened in the past, but also what will happen in the future.

Example of such state defined by future:
1. Observable operator models
2. Predictive state representation
3. Multiplicity automata

#### State-free modelling of temporal system
Engineering view on "systems": we map input directly to output where we do not care how the system functions.

Continuous vs digital-time signals and systems
Before computers: continuous to continuous
Now: we take continuous signals, sample them and get digital signals.

Different times of filters that perform the sampling:
1. Linear filter (very powerful)
	- Linear systems: possible to prove correct in some capacity. (currently) not possible for NNs.
2. Time invariant systems
	- As time goes on, local transforms do not change
	- Together: "LTI" - linear time-invariant systems
3. Causal systems
	- Only depends on past things, and not future things
4. BIBO stable

Linear systems: frequency-domain analysis (Fourier or Laplace transforms). That is why linear systems are so powerful in signal processing.

##### Context-free grammars (CFG)
Defined by: a finite set of variables, a start variable, a finite set of terminals, a finite set of production rules. A CFG defines a language i.e. a set of finite symbol sequences (words) over the terminals through derivations. (Tree form)

- Grammars are a prime tool to describe long-range dependencies in symbol sequences

##### Qualitative theory of DS: Attractors, bifurcations, chaos
- Example in the slides (Title: Our tutorial example)
- Vector fields to describe what is happening in such a system
- The vector field will guide the agent, to give it a trajectory
- Plotting several trajectories with different starting points gives a phase portrait.
- Bifurcations: when a control parameter passes through a critical value, the phase portrait changes its nature.

### Dynamics system 4
[Link](https://www.youtube.com/watch?v=9M46gfEQLh0)

Phase transition vs Bifurcations (not the same thing)
Most famous phase transition: water ice to liquid at 0C, liquid to gas at 100C.

#### Difference between PT and bifurcations
- Maths is different
- PT is only defined when we go to infinite
- PT free energy suffers a discontinuity when we change the control parameters
- There are connections but are not the same thing.

#### Attractors mathematically defined
- Attractors are more interesting to understand the world compared to repellors
- Repellors are mathematically there, but not observable
- We are all attractors, otherwise, we would not exist. This is why attractors are interesting.

Topological dynamics (a more abstract way to encapsulate all dynamics)
Tries to abstract away from geometry. What does it mean to be close?
The only thing we care about here is to define what it means to be close, neighbourhood

Topological dynamics are nice because we can define attractors formally.
Attractors, A, are a subset of a topological space if the
- If you are in A, you will stay in A no matter what.
- If you are close enough to A, you will be attracted to it, there is no escape.

#### Connections between symbols and attractors
In our brain, we have "objects"
Our brain is a high dynamical system
One natural modelling approach: symbols = attractors in neural dynamics

##### Other modelling approaches
1. discrete regions in neural state space
2. individual neurons or areas
3. dimensions/subspaces in neural state space
4. saddle nodes
5. conceptors

##### Symbols as attractors
Pros:
- temporal stability
- Hopfield networks
Con:
- inherent conflict: attractors by definition capture neural trajectory forever, while thinking transits from concept to concept
- attempts to resolve this con: neural noise, generalized / modified attractor concepts / attractor-like phenomena

Opinion: We do not know how to do this in high dimensional systems. All our examples are from low dimensional systems.

#### Non-autonomous dynamical systems
We have input, hence a non-autonomous system.
Def: x(n+1) = T\_n(x(n)) (we have input)
Example: an RNN trained on video input stream to predict the next symbol using grammars.

