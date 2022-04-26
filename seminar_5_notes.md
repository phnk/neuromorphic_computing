# Seminar 5

# Homework (Hardware)
## [Video #1](https://www.youtube.com/watch?v=u9pulFlavIM)
* What is Von Neumann Architecture?
	- See previous lecture notes.
* What do we want to achieve with Neuromorphic hardware?
	- Use analog curcuits to mimic neuromorphic architectures.
* Memoristive device
	- Material based memory devices where the resistance can be programmed and stored
* AHAH computing
* Akita processor uses 90-99% less power
* Many applications
* What is the concept of Neuromorphic Computing?
	- Mimics the brain
	- Uses synapses, neurons, etc.
	- Sends information depending on spikes
	- Changes because of stimlui
* Memristors
	- Store and transmit information
	- Can store a range of values.
	- Allow it to mimic the defering connection strengths of various synapses
	- Save energy because it does not do work when not spiking
	- Local rather than global
* Biological vs Silicon Neurons
	- Voltage
	- Energy
	- Total energy
* Computers lacking in two main areas
	- Transfer learning
	- Machine reasoning
* NC high potential for space operations

# [Video #2](https://tube.switch.ch/videos/db393d1d)
## Giacomo Indiveri
* One term, multiple communities
	- Neuromorphic Engineering
	- Neuromorphic Computing
	- Neuromorphic Devices
* "Tries to emulate physics of cells with the physics of transistors"
* Majority carries
* Subthreshold analog circuts Ids/V plot recembles the Test pulse potential plots (NA conductance and K conductance)
* Synaptic dynamics
	- The circut mimics the real synapse mechanics with a slow-down
	- Large difference between the nA (wasted energy?)
	- Discontinous (first order diff eq.)
	- Can get it smooth using two circuits. 1 negative and 1 positive.
* Spiking neural network chips
	- Mix of both digital and analog signals (input output digital, rest analog)
	- Analog subthreshold circuits
	- Slow temportal, non-linear dynamics
	- Massively parallel operation
	- Inhomogeneous, imprceise and noisy
	- Adaptation and learning at multiple time scales
	- Fault tolerant and mismatch insensitive by design
	- Fast asnchronous digital routing circuits
	- Re-programmable network topology and connectivity
	- We can make these chips, can we make them do something productive? How do we configure the parameters?
* Look at other brains
	- Bee brain
* Neuromorphic agents
	- Exploit physical space
	- Let time represent itself
	- Use both analog and digital computing elements
	- Minimize wiring (Maximize local connectivity)
	- Exploit the non-linearities and temporal dynamics
	- Optimize for real-time interaction
	- Match the network dnyamics to the spatio-termonal signal of intereset
	- Exploit noise to do operations
	- Re-use computational primites to implement sensory processing, motor control, and cognitive computing
* Neuromorphic processors (NP)
	- Pros: Low latency, Ultra low power
	- Cons: Limited resolution, High variability, noisy
* What are NP good for?
	- Real-time sensory-motor processing
	- Sensory-fusion and on-line classification
	- Low-latency decision making
* What are they bad at?
	- High accuracy pattern recognition
	- High precision number crunching
	- Batch processing of data sets (Loading from disc will have a drastic energy consumption)
* How do we program a NP?
	- Two ideas: Data-driven or rule-based
	- Data-driven: Configure the network structure and train
	- Rule-based: Define computational primitives. Compose multiple primitives for context dependent processing
* Conclusion
	- Combine multiple disciplines
	- Emulate the biophyisics of neural systems
	- Time represent itself
	- Implement robust computation in autonomous agents that produces cognitive behaviour

## Steve Furber
* Discussed history of computing/CPU
* SpiNNaker project
* Chip resources
	- Router (routing tables, spike packet routing)
	- Ram port (synapse states, activity logs)
	- Indstruction memory (run-time kernel, application callbacks)
	- Data memory (kernel state, neuron state, stack and heap)
	- Processor (neuron and synapse state computations)
* Applications
	- Computational Neuroscience
	- Theoretical Neuroscience
	- Neurorobotics

## Bernabé Linares-Barranco
* Vision and vision sensing and processing
* Human brain
	1. Image flashes
	2. Retina
	3. LGN
	3. Virtual Cortex
	4. PIT
	5. AIT
	6. PFC (before hitting this point we already see the human face)
	7. PMC
	8. MC
	9. Spinal Cord
	10. Pressing the button
* Takes <150ms
* All neurons only have time to fire a single spike in this entire cycle
* Means high power efficiency and coding efficiency
* Dynamic Vision sensors
	- Already know this information
* Convolution on such type of sensors
	- Gets x, y events from the sensor
	- Puts into a matrix of LIF neurons
	- When the entire things reaches some threshold, it spikes
	- Applications: Use to detect the colours of cards. 
	- Uses event-based sensing to do it

## Kwabena Boahen
* 3D silicon brains
* Described GPT-3
* Described attention
* Attention: If Query \approx Key, then Input -> Value
* Bilinear transformation
* To decrease error a small amount, we need to increase the compute drastically
* "On-device AI"
	- Energy-saving
	- Personalized
	- Real-time
	- Secure
* Transformer Net
	- Size = depth x width
	- Dense local connections
	- Sparse global projections
* "Compute in memory"
	- Energy scales as size^2
* Memory has shifted from 2D to 3D where we stack memory in layers now
	- We can fit GPT-3 on these.
* 3D integration
	- Energy scales as size^1.5 for depth << width
	- "Dark silicon"
* Making Deep-Net's dense signals sparse
	- When its input is swept, a neruon switches once, not many times
	- Binary vs Unary
* 3D + Unary
	- Energy scales as size for depth << width
	- Thermally viable
* Move from cloud to edge
	- From kilowatts to watts

# [Video #3](https://www.youtube.com/watch?v=jhQgElvtb1s)
* Proving the value of spiking neuromorphic hardware
* We want to make chips that solve real world problems
* Need new architecture that is efficient
* Where does the brains efficiency and capability come from? How can we realise this in hardware?
* SNN is the simpelest addition to ANNs that greatly broadens the algorithimtic space we are capable of computing.
* Loihi
	- Event based processing pushed all the way to the transistors
	- Async
* The challenge: SNN algorithm discrovery
	- Deep learning derivied approaches
	- New ideas guided by neuroscience
	- Mathematically Formalized
* DNN-to-SNN conversion
	- Keyword spotting
	- 5-10x lower energy
	- Caveats: Reduced accuracy, batchsize=1
* Searching small world networks with loihi
	- Dijktras
* Solving constraint satifaction problems
	- 4ms time to solution on 4 coloring of world map and soduku
* Research frontier
	- Inference and learning of sparse feature representations
	- Video and speech recognition
	- Event-based camera processing
	- Chemosensing
	- Robotics
	- Adaptive dynamic control
	- Anomaly detection for security and industrial monitoring
	- Optimization: Constraint satisfaction, QUBO, contex optimization
	- Autonomy: SLAM, planning, closed-loop behaviour

# Seminar
* Brain-inspired computing needs a master plan
	- Need to simulate same amount of neruons as the brain in a ANN u need MWs of power.
	- Motivations why we are doing what we are doing.
	- Article in Onedrive with annotations.

* Intel Neuromorphic Computing Workshop in April 2022

* [Quantum x NC](https://www.youtube.com/watch?v=IP_GmTKYlsc)

## Hardware Seminar (Mattias)
* "Silicon neurons and synapses"
* Use analog circuts to model the neurondynamics
* Main blocks of silicon-neurons "Soma"
	1. Temporal Integration
	2. Spike generation
	3. Refractory mechanism
	4. Spike ferq adapation
	5. Spike threshold adapation
* Wide set of circuit design styles (or decisions)
	- When are each of these used? When and why are x better than y?
	- We can accelerate all of these styles?
* Placicity
	- On chip learning
	- We need a circut block to update weights
* SoTA
	- CMOS vs FDSOI?
	- What is the hurdle to experimental SoTA?

## Event Triggered Extended Kalman Filter (Moumita)
* Fundamentals of State estimation
* System Dynamics
	- x* = Ax + Bu + Gw
	- y = Cx + v
* Estimator Dynamics
* Kalman filter
* Event triggered Estimation
	- Smart Sensor
	- Event-tiggered Scheduler
	- We process when there is a spike

## A neuromorphic colution to CSPs (Kim)
* Constraint satisfaction problem
* Common solutions given by variable elimitation
* Focus on 4 principles
* Motifs from the paper
	- Winner-take-all for variables
	- OR for constraints
	- Principal neurons: variables to the system. does main computation
	- Auxillary neurons: constraints
	- Power of motifs: Modularity
* Network states
	- x(t) is the network state
	- p(t) is the unique stationary distribution
	- Energy function E(X) = ....
* Stochastic input
	- See slide for information
	- The search function can take a shortcut and bypass high energy states to find the low energy states
* Temperature
	- If you get a good solution it scales up the landscape so its harder to escape the good solution


