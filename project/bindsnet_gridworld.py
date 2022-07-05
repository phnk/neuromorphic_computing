import gym
import gym_gridworld
import torch
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.encoding import bernoulli, repeat, single
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=5 * 5, shape=[1, 1, 1, 5, 5], traces=True)
middle = LIFNodes(n=100, traces=True)
out = LIFNodes(n=4, refrac=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the Breakout environment.
environment = GymEnvironment("GridWorld-v0")
environment.reset()

# Build pipeline from specified components.
environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=repeat,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=5,
    delta=1,
#    plot_interval=1,
#    render_interval=1,
)

reward_list = []

def run_pipeline(pipeline, episode_count):
    for i in range(episode_count):
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False
        for _ in range(1000):
            result = pipeline.env_step()
            new_result = (result[0], result[1] + 0.0, result[2], result[3])
            pipeline.step(new_result)

            reward = new_result[1]
            total_reward += reward

            if new_result[2]:
                print("done")
                pipeline.reset_state_variables()

        reward_list.append(total_reward)
        print(f"Episode {i} resulted in total reward of: {total_reward}")


print("Training: ")
run_pipeline(environment_pipeline, episode_count=100)
print(reward_list)
plt.plot(reward_list)
plt.savefig("asdf.png")

# stop MSTDP
#environment_pipeline.network.learning = False

#print("Testing: ")
#run_pipeline(environment_pipeline, episode_count=100)
