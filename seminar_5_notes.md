# Seminar 5

# Homework (Hardware)
[Video #1](https://www.youtube.com/watch?v=u9pulFlavIM)
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

[Video #2](https://tube.switch.ch/videos/db393d1d)
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

Continue at 1h2m
