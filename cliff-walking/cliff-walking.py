import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

"""
Write a program to solve a cliff walking problem from a gymnasium environment.
The program should use reinforcement learning.
"""

env = gym.make('CliffWalking-v0')
# get the number of states and actions
N_STATES = env.observation_space.n
N_ACTIONS = env.action_space.n


# define the Q-learning function
def q_learning(alpha, gamma, num_episodes):
    Q = np.zeros((N_STATES, N_ACTIONS))
    learning_curve = np.zeros(num_episodes)

    state = env.reset()[0]
    for i in range(num_episodes):

        done = False
        episode_reward = 0

        while not done:
            # Choose action using epsilon-greedy policy
            epsilon = 1.0 / (i + 1)
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            # Take action and observe the next state and reward
            observation, reward, terminated, truncated, info = env.step(action)

            # Print
            print(f'Episode {i}: observation={observation}, action={action}, state={state}, reward={reward};')

            # Update Q
            Q[state, action] += alpha * (reward + gamma * np.max(Q[observation]) - Q[state, action])

            state = observation
            done = terminated

            episode_reward += reward

        learning_curve[i] = episode_reward

    np.save('learning_curve.npy', learning_curve)

    learning_curve = np.load('learning_curve.npy')
    plt.plot(learning_curve)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')

    return Q


# perform Q-learning
Q = q_learning(alpha=0.5, gamma=0.9, num_episodes=5)

total_reward = 0
state = env.reset()[0]
done = False

while not done:
    action = np.argmax(Q[state])
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    state = observation
    if terminated or truncated:
        done = True

print("Total reward:", total_reward)


