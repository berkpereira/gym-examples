import gym
import gym_examples
env = gym.make('gym_examples/GridWorld-v0', size=4, render_mode='human')
env.action_space.seed(42)

observation, info = env.reset(seed=42)

while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        env.reset()

env.close()
