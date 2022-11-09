import gym
import follow_the_leader_gym
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import load_results, ts2xy

from bindsnet.encoding import *
from bindsnet.environment import GymEnvironment
from bindsnet.network import Network
from bindsnet.network.nodes import (
    AbstractInput,
    IFNodes,
    Input,
    IzhikevichNodes,
    LIFNodes,
    Nodes,
)
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import *


def plot_results(log_folder, title="training phase"):

    def moving_average(values, window): 
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")

    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.title(title)
    plt.savefig("fig_after_training.png")

# Init the network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.features_extractor = nn.Flatten(start_dim=-1, end_dim=-1)
        self.q_net = nn.Sequential(*[nn.Linear(9, 1000), nn.ReLU(), nn.Linear(1000, 2)])

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.q_net(x)
        return x

if __name__ == "__main__":
    q_net = QNetwork()
    print(q_net)

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Initialize env
    log_dir = "logs/"

    env = gym.make("Follow-The-Leader-v0")
    env = Monitor(env, log_dir)

    new_logger = configure(log_dir, ["stdout", "csv"])

    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[1000])

    # hyperparams from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml#L52
    model = DQN("MlpPolicy", 
                env,
                policy_kwargs=policy_kwargs,
                verbose=2)

    timesteps = 1e0

    model.set_logger(new_logger)

    model.learn(total_timesteps=timesteps, log_interval=4)

    sd = model.policy.q_net.state_dict()
    #print(sd)
    #model.save("dqn_follow_trained")
    #plot_results(log_dir)

    model = DQN.load("dqn_follow_trained", env=env)

    for i in range(3):
        obs = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                print(total_reward)
                break

    q_net.load_state_dict(sd)

    SNN = Network(dt=1.0)

    inpt = Input(n=9, traces=False)
    middle = LIFNodes(n=1000, refract=0, traces=True, thresh=-52.0, rest=-65.0)
    readout = LIFNodes(n=2, refrac=0, traces=True, thresh=-57.0, rest=-65.0)
    layers = {"X": inpt, "M": middle, "R": readout}

    inpt_middle = Connection(
            source=layers["X"],
            target=layers["M"],
            w=torch.transpose(q_net.q_net[0].weight, 0, 1) * 57.68 
            )

    middle_out = Connection(
            source=layers["M"],
            target=layers["R"],
            w=torch.transpose(q_net.q_net[2].weight, 0, 1) * 77.68 
            )

    SNN.add_layer(inpt, name="input layer")
    SNN.add_layer(middle, name="hidden layer")
    SNN.add_layer(readout, name="output layer")
    SNN.add_connection(inpt_middle, source="input layer", target="hidden layer")
    SNN.add_connection(middle_out, source="hidden layer", target="output layer")

    snn_env = GymEnvironment("Follow-The-Leader-v0", clip_rewards=False)
    snn_env.reset()

    pipeline = EnvironmentPipeline(
        SNN,
        snn_env,
        encoding=bernoulli,
        encode_factor=50,
        action_function=select_highest,
        output="output layer",
        reset_output_spikes=True,
        time=1000,
        device=device,
    )

    reward_list = []
    for i in range(3):
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False
        pipeline.env.step(1)
        pipeline.env.step(1)
        while not is_done:
            result = pipeline.env_step()
            is_done = result[2]
            if is_done:
                reward_list.append(total_reward)
                print(total_reward)
                break

            pipeline.step(result)

            reward = result[1]
            total_reward += reward

    print(reward_list)

    """
    obs = env.reset()

    # Enjoy
    for _ in range(500):
        action, _ = model.predict(obs)
        print(action)

        obs, reward, done, info = env.step(action)
        env.render()

    env.close()
    """
