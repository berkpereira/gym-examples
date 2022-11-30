import gymnasium as gym
import gym_examples
import pygame

def grid_env_constructor(grid_size):
    env = gym.make('gym_examples/GridWorld-v0', size=grid_size, render_mode='human')
    env.action_space.seed(42)
    return env

def run_environment(grid_size, policy):
    env = grid_env_constructor(grid_size)
    observation, info = env.reset(seed=42)
    print('New episode!')
    while True:
        action = policy(env)
        
        
        # somewhere around here we're going to call the policy_evaluation function
        
        
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'action: {action}    state: {observation["agent"]}    target: {observation["target"]}')
        print(f'reward: {reward}')
        print()

        if terminated or truncated:
            env.close()
            #observation, info = env.reset()
            print("Episode finished!")
            #print("""
            #""" * 5)
            #print("New episode!")


def test_policy(env):
    return env.action_space.sample()


class ValueFunction():
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.current_value = np.zeros([grid_size, grid_size])

        

    

    

def policy_evaluation(grid_size):
    value_function = np.zeros([grid_size, grid_size])

run_environment(grid_size = 5, policy = test_policy)