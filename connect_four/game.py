import time

import numpy as np
from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import OrderEnforcingWrapper
from tqdm import tqdm
from q_learning import QLearner, plot_learning_curve


class Game:
    def __init__(
            self,
            env: OrderEnforcingWrapper,
            player_0: QLearner,
            player_1: QLearner,
    ) -> None:
        self.env = env
        self.player_0: QLearner = player_0
        self.player_1: QLearner = player_1

    def train(self, epoch: int, verbose: bool = True) -> None:

        # create and save learning curves
        learning_curve_0 = np.zeros(epoch)
        learning_curve_1 = np.zeros(epoch)

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0

        for i in tqdm(range(epoch)):
            was_draw: bool = False
            self.env.reset()
            for agent in self.env.agent_iter():
                last_observation, reward, termination, truncation, info = self.env.last()
                if termination:
                    if reward == 1 and agent == "player_0":
                        nb_wins_agent_1 += 1
                        learning_curve_0[i] = 1
                        learning_curve_1[i] = -1
                    elif reward == 1 and agent == "player_1":
                        nb_wins_agent_2 += 1
                        learning_curve_0[i] = -1
                        learning_curve_1[i] = 1
                    elif reward == 0 and agent == "player_0":
                        was_draw = True
                    elif reward == 0 and agent == "player_1" and was_draw:
                        nb_draws += 1
                        learning_curve_0[i] = 0
                        learning_curve_1[i] = 0

                    if i % 500 == 0 and verbose and i != 0 and agent == "player_0":
                        print(
                            f"\nAgent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}, "
                            f"Ratio: {100 * (nb_wins_agent_1 / (nb_wins_agent_2 + nb_wins_agent_1)) : .2f}"
                        )

                    self.env.step(None)
                elif truncation:
                    if verbose:
                        print("Truncated")
                else:  # we update the actor and critic networks weights every steps
                    if agent == "player_0":
                        action = self.player_0.get_action(last_observation)
                    else:
                        action = self.player_1.get_action(last_observation)

                    while last_observation['action_mask'][action] == 0:
                        if agent == "player_0":
                            action = self.player_0.get_action(last_observation)
                        else:
                            action = self.player_1.get_action(last_observation)

                    self.env.step(action)
                    observation, reward, termination, truncation, info = self.env.last()

                    if agent == "player_0":
                        self.player_0.update(
                            last_observation, action, reward, termination, observation
                        )
                    else:
                        self.player_1.update(
                            last_observation, action, reward, termination, observation
                        )

        plot_learning_curve(learning_curve_0, "player_0")
        plot_learning_curve(learning_curve_1, "player_1")

    def eval(self, nb_eval: int, verbose: int = 0):

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0
        i: int = 0
        while (nb_wins_agent_1 + nb_wins_agent_2 + nb_draws) < nb_eval:
            self.env.reset()
            was_draw: bool = False
            for agent in self.env.agent_iter():
                last_observation, reward, termination, truncation, info = self.env.last()
                if termination:
                    if reward == 1 and agent == "player_0":
                        nb_wins_agent_1 += 1
                        i += 1
                    elif reward == 1 and agent == "player_1":
                        nb_wins_agent_2 += 1
                        i += 1
                    elif reward == 0 and agent == "player_0":
                        was_draw = True
                    elif reward == 0 and agent == "player_1" and was_draw:
                        nb_draws += 1
                        i += 1

                    self.env.step(None)
                elif truncation:
                    if verbose:
                        print("Truncated")
                else:
                    if agent == "player_0":
                        action = self.player_0.get_action(last_observation)
                    else:
                        action = self.player_1.get_action(last_observation)

                    while last_observation['action_mask'][action] == 0:
                        if agent == "player_0":
                            action = self.player_0.get_action(last_observation)
                        else:
                            action = self.player_1.get_action(last_observation)

                    self.env.step(action)

        print(f"Agent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}")

    def watch(self):
        env = self.env
        self.env = connect_four_v3.env(render_mode="human")
        self.env.reset()
        for agent in self.env.agent_iter():
            last_observation, reward, termination, truncation, info = self.env.last()
            if termination:
                print(f"Termination ({agent}), Reward: {reward}")
                self.env.step(None)
            elif truncation:
                print("Truncated")
            else:
                if agent == "player_0":
                    action = self.player_0.get_action(last_observation)
                else:
                    action = self.player_1.get_action(last_observation)

                while last_observation['action_mask'][action] == 0:
                    if agent == "player_0":
                        action = self.player_0.get_action(last_observation)
                    else:
                        action = self.player_1.get_action(last_observation)
                self.env.step(action)
                self.env.render()
                time.sleep(2)
        self.env = env
