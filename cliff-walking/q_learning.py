import numpy as np
import matplotlib.pyplot as plt


# define the Q-learning function
def q_learning(env, n_states, n_actions, epsilon, alpha, gamma, n_episodes):
    Q = np.zeros((n_states, n_actions))
    learning_curve = np.zeros(n_episodes)

    total_reward = 0
    for episode in range(n_episodes):
        state = env.reset()[0]
        terminated, truncated = False, False
        episode_reward = 0

        while not terminated and not truncated:
            # Choose action
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            # Take action and observe the next state and reward
            observation, reward, terminated, truncated, info = env.step(action)

            # Print
            print(f'Episode {episode}: observation={observation}, action={action}, state={state}, reward={reward};')

            # Update Q
            Q[state, action] = alpha * (reward + gamma * np.max(Q[observation])) + (1 - alpha) * Q[state, action]
            state = observation

            episode_reward += reward
            total_reward += reward

        learning_curve[episode] = episode_reward

    np.save('learning_curve.npy', learning_curve)

    learning_curve = np.load('learning_curve.npy')
    plt.plot(learning_curve)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')

    return Q, total_reward
