import gymnasium as gym
import numpy as np
from q_learning import q_learning, plot

"""
Program solving a cliff walking problem from a gymnasium environment using Q-learning.
"""

env = gym.make('CliffWalking-v0')

# get the number of states and actions
N_STATES = env.observation_space.n  # number of states in the environment (48)
N_ACTIONS = env.action_space.n  # number of actions in the environment (4)
ALPHA = 0.5  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.1  # epsilon-greedy policy
N_EPISODES = 500  # number of episodes

# perform Q-learning
Q, learning_curve = q_learning(env=env, n_states=N_STATES, n_actions=N_ACTIONS, epsilon=EPSILON, alpha=ALPHA,
                               gamma=GAMMA, n_episodes=N_EPISODES)

# plot the learning curve
plot(learning_curve)

# evaluate the learned policy
evaluate_episodes = 50
total_reward = 0
for episode in range(evaluate_episodes):
    terminated, truncated = False, False
    episode_reward = 0
    state = env.reset()[0]

    while not terminated and not truncated:
        action = np.argmax(Q[state])
        # Take the chosen action
        state, reward, terminated, truncated, info = env.step(action)
        # Update rewards
        episode_reward += reward
        total_reward += reward

    print(f'Episode numer {episode + 1}: reward={episode_reward}')

print(f'Average reward over {evaluate_episodes} episodes: {total_reward / evaluate_episodes}')
