import gym
import follow_the_leader_gym
import torch
from torch import nn
from tqdm.auto import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

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

# Callbacks
class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

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

q_net = QNetwork()
print(q_net)

device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

# Initialize env
log_dir = "logs"
env = gym.make("Follow-The-Leader-v0")
env = Monitor(env, log_dir)

# Train
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[256, 256])
policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[1000])

# hyperparams from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml#L52
model = DQN("MlpPolicy", 
            env,
            policy_kwargs=policy_kwargs)

timesteps = 1e6

#with ProgressBarManager(timesteps) as callback:
#    model.learn(total_timesteps=timesteps, callback=callback)

sd = model.policy.q_net.state_dict()
print(sd)
#model.save("dqn_follow_trained")

model = DQN.load("dqn_follow_trained", env=env)

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


snn_env = GymEnvironment("Follow-The-Leader-v0")
snn_env.reset()


pipeline = EnvironmentPipeline(
    SNN,
    snn_env,
    encoding=bernoulli,
    encode_factor=50,
    action_function=select_highest,
    percent_of_random_action=0.05,
    random_action_after=5,
    output="output layer",
    reset_output_spikes=True,
    time=500,
    #overlay_input=4,
    history_length=1,
    plot_interval=1,
    #render_interval=1,
    device=device,
)

for i in range(1):
    total_reward = 0
    pipeline.reset_state_variables()
    is_done = False
    pipeline.env.step(1)
    pipeline.env.step(1)
    while not is_done:
        result = pipeline.env_step()
        is_done = result[2]
        if is_done:
            print(result)
            break

        pipeline.step(result)

        reward = result[1]
        total_reward += reward


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
