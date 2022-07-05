import gym
import gym_gridworld


action_to_direction = {
    0 : "down", 
    1 : "right", 
    2 : "up", 
    3 : "left" 
}

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
