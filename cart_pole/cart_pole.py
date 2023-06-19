import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

tmp_path = "logs/"
writer = SummaryWriter()


class TensorboardCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        self.writer.add_scalar("reward", self.locals["rewards"][0], self.num_timesteps)
        return True


def run_experiment(env_name, hyperparameters, num_runs=10, total_timesteps=50000):
    print(f"Running experiment for {env_name} with hyperparameters {hyperparameters}")
    rewards_all_runs = []
    stds_all_runs = []

    for run in range(num_runs):
        # set up logger
        new_logger = configure(tmp_path, ["csv"])
        vec_env = make_vec_env(env_name, n_envs=1)
        model = PPO("MlpPolicy", vec_env, verbose=1, **hyperparameters)
        callback = TensorboardCallback(writer, verbose=0)
        model.set_logger(new_logger)
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # read the csv file as pandas dataframe
        df = pd.read_csv(tmp_path + "progress.csv", sep=",")
        # get the column with the mean reward per episode
        rewards = df["rollout/ep_rew_mean"].to_numpy()
        time_elapsed = df["time/time_elapsed"].to_numpy().max()
        print(f"Run {run + 1} took {time_elapsed} seconds")
        # calculate mean reward and std
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        rewards_all_runs.append(mean_reward)
        stds_all_runs.append(std_reward)

    return rewards_all_runs, stds_all_runs


# Define multiple sets of hyperparameters to compare
hyperparameters_list = [
    {"learning_rate": 0.001, "gamma": 0.99},
    {"learning_rate": 0.01, "gamma": 0.99},
    {"learning_rate": 0.01, "gamma": 0.95}
]

env_name = "CartPole-v1"
num_runs = 10
total_timesteps = 50000

# Run experiments for each set of hyperparameters
learning_curves = []
for hyperparameters in hyperparameters_list:
    rewards_mean, rewards_std = run_experiment(env_name, hyperparameters, num_runs, total_timesteps)
    learning_curves.append((rewards_mean, rewards_std))

# Plot the learning curves
fig, axs = plt.subplots(len(hyperparameters_list), figsize=(8, 6))
for i, (params, (rewards, stds)) in enumerate(zip(hyperparameters_list, learning_curves)):
    axs[i].plot(rewards, label='Reward')
    axs[i].fill_between(range(len(rewards)), np.array(rewards) - np.array(stds), np.array(rewards) + np.array(stds),
                        alpha=0.3)
    axs[i].set_title(f'Hyperparams: {params}')
    axs[i].set_xlabel('Step/Episode')
    axs[i].set_ylabel('Reward')
    axs[i].legend()
plt.tight_layout()
plt.show()

# Close the writer
writer.close()
