import gymnasium as gym
import numpy as np
from q_learning import q_learning

"""
Write a program to solve a cliff walking problem from a gymnasium environment.
The program should use reinforcement learning.
"""

env = gym.make('CliffWalking-v0')

# get the number of states and actions
N_STATES = env.observation_space.n  # number of states in the environment (48)
N_ACTIONS = env.action_space.n  # number of actions in the environment (4)
ALPHA = 0.5  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.1  # epsilon-greedy policy
N_EPISODES = 100  # number of episodes

# perform Q-learning
Q, total_reward = q_learning(env=env, n_states=N_STATES, n_actions=N_ACTIONS, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                             n_episodes=N_EPISODES)

# evaluate the learned policy
evaluate_episodes = 10

for episode in range(evaluate_episodes):
    state = env.reset()[0]
    terminated, truncated = False, False
    total_rewards = 0

    while not terminated and not truncated:
        action = np.argmax(Q[state])
        # Take the chosen action and observe the outcome
        state, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward

    print(f'Episode {episode + 1}: total_rewards={total_rewards}')
