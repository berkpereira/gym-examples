import gym
import gym_examples
env = gym.make('gym_examples/GridWorld-v0', size=10, render_mode='human')
env.action_space.seed(42)

observation, info = env.reset(seed=42)

"""
while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(env.action_space)

    if terminated or truncated:
        observation, info = env.reset()

"""


observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
