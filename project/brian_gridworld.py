import gym
import gym_gridworld


action_to_direction = {
    0 : "down", 
    1 : "right", 
    2 : "up", 
    3 : "left" 
}


# see: https://github.com/brian-team/brian2/issues/980#issuecomment-404440433
# see: https://github.com/BindsNET/bindsnet#background

# Input "layer"
# we have an obs. This obs is a n x n matrix containing 3 different values, 0, 1 and 2.
# do we flatten? how do we "fit this" into the input. 1 neuron per input? seems useless.

# define eqs for the NeuronGroup and threshold
# leaky integrate-and-fire
tau = 10*ms
El = -70*mV

# see: https://github.com/brian-team/brian2/blob/master/docs_sphinx/introduction/brian1_to_2/library.rst
eqs = '''
    dvm/dt = ((El - vm) + I)/tau : volt
    I : volt
'''

# define Synapse connection between Input and NeuronGroup

# attach a spike monitor

# Do a run

# plot results


env = gym.make("GridWorld-v0", size=5)
obs, info = env.reset(seed=42)
for _ in range(1000):
#    env.render()
    action = env.action_space.sample()
#    print(action_to_direction[action])
    obs, reward, done, info = env.step(action)

    if done:
        obs, info = env.reset(seed=42)
env.close()
