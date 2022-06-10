# Seminar 9

## Homeowork
### [What is an event-driven sensors?](https://tube.switch.ch/videos/5a6c3a74)
#### Tobi Delbruck
* Silicon retina - Mimic human eye.
* Started with Mahowald/Mead doing something similar to edge detection with analog circuts.
	- High mismatch, hard to use. Basically no improvement over normal image sensors.
* DVS (Dynamic Vision Sensor)
	- A pixel detects brightness change events +- delta log I.
* DVS output change rather than all information always.
* Demo with the DVS

#### Angel Jimenez-Fernadez
* What is an event-driven audio sensor?
	- Embedded auditory device inspired by biological inner ear models
	- For real-time operation
	- Work like an audio pre-processing layer
* Many applications, such as speaker identification
* For biological explanation: "Human and Machine hearing"
* Many different types of sensor exist now
	- DAS (Dynamic Audio Sensor)
	- CochLP
	- CAR-Lite
	- Neuromorphic Auditory Sensor (NAS)
* Event-based audio sensors are alive
* New emerging designs
* Large field of research
* Implementing biologically plausable can aid people with hearing imperament.
* Video example

#### Chiara Bartolozzi
* Neuromorphic Touch
* Why is it important in biology: we can learn properties, how to handle and manipulate objects.
* Why is it important for robotics: Grasping force, exploration, balancing, safety, etc.
* Another important domain: Prosthetics. Same as why its important for biology.
* How do we move from biology to arificial touch?
	- How does touch work in biology?
	- How is the information encoded?
	- How can we design sensors?
	- How can we use tactile information in robotics?
* Two approahes: soft and hard.
* Soft neuromorphic approach
	- Clock front-end and event-driven encoding
* Hard neuromorphic approach
	- Asynchronous readout of transducer

#### Michael Schmuker
* What is olfaction? Smell
* Why is olfaction?
	- Biology: Locating food, mating partners, predators.
	- Technology: gas-based navigation, gas leaks etc.
* Problem: Olfaction does not travel in a straight line. Its convoluted all over the place.
* Decreases over time, hence a set of sensors in a row will give high variance.
	- However, this allows us to calculate the distance from source.
* Working principle of gas sensors.
	- Hot plate, that heats metal oxide that connects two electrode. 
	- Gas hits metal oxide, the resistance changes and we can measure the change.
* From this we just trace the voltage, apply deadband sampling and we obtain events. We now how a neuromorphic gas sensors.
* Practical issue: MOx sensors are slow.
	- We can speed it up with signal processing. Apply signal processing to recover the rapid onset (Kalman filter).
* Event-based gans sensing is a natural fit to turbulent dispersal properties of odorants.
* Information incoded in turbulent plumes is caputred by events.
* Gas discirimination is possible in the event domain.


### [HOW TO PROCESS INFORMATION FROM EVENT-DRIVEN SENSORS?](https://tube.switch.ch/videos/kTa2YuDI3n)

#### Ryad Benosman
* An introduction to event based vision
* Event-based cameras have become a commodity
	- Wide set of startups and established companies within the space
* The general sensing schema is:
	1. Sensor
	2. Data structure
	3. Process
	4. Get some result
* Event based cameras is different
	- We want to do some aplitude sampling
	- The information is only sent on change
	- This is a sparse coding
	- Time is the most valuable information
* Computer vision: The impossible trade off between power and frame rate
	- Oversampling or undersampling
	- High power and high latency
* How can we move away from frames even in event based vision?
	- Update previous result with the change
* Event-based cameras solve low power and low letancy
* ***Any problems can be translated into an icremental approach***
* Any form of temporal integration of events loses all advantages of these sensors (grey level, reconstruction or frame generation)

#### Jonathan Tapson
* How do we process event based sensors on event processing?
* Why neuromorphic computing?
	- Low power
	- Low latancy
	- Micro level parallelism
	- "Hyper-resolution"
* Where are these precived advantages derived from?
	- Spikes, events, sparsity, asynchrony, etc etc.
* Sparsity is everywhere and can be exploited everywhere
* Event-based sensing is very good for high correlated low time different data

#### Bertram Shi
* "Velocity is the main feature. If nothing changes nothing is sent"
* We are not simply passive, but we observe
	- Eye movements
* Control of eye movements
	- Open loop: stimuli indepdenant
	- Closed loop: stimuli dependant
* Small eye movements critical for perception
* Many event representation. Bertams favorite is: Time surfeces.
* Underlying world is slow temporal
* Closed loop: Optokinetic Reflex
	- We can stabilize our eyes (Such as looking out of a train. We are moving but the background looks staionary)
* It is time to think about how to control our event-based sensors to structure the generated events to facilitate perception


## Seminars 
