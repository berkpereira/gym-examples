import gym
import gym_examples

def grid_env_constructor(grid_size):
    env = gym.make('gym_examples/GridWorld-v0', size=grid_size, render_mode='human')
    env.action_space.seed(42)
    return env


env = grid_env_constructor(4)
# in the below, do env.rest(seed=42) to get repeated initialisations
observation, info = env.reset()

def test_policy(env):
    #return env.action_space.sample()
    return 1

def policy_evaluation(policy):
    
    
    while True:
        action = policy(env)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'action: {action}    state: {observation["agent"]}    target: {observation["target"]}')

        if terminated or truncated:
            env.reset()

    #env.close()


policy_evaluation(test_policy)