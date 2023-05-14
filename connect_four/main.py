from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import OrderEnforcingWrapper

from q_learning import QLearner
from game import Game


def main(epoch: int, eval_epoch: int, verbose: bool = True) -> None:

    env: OrderEnforcingWrapper = connect_four_v3.env()
    space = env.action_space
    player_0 = QLearner(space, "player_0", name="player_0")
    player_1 = QLearner(space, "player_1", name="player_1")
    game = Game(env, player_0, player_1)

    game.train(epoch=epoch, verbose=verbose)
    game.eval(nb_eval=eval_epoch, verbose=verbose)
    game.watch()


if __name__ == "__main__":
    N_EPOCH = 40
    N_EVAL = 5
    main(epoch=N_EPOCH, eval_epoch=N_EVAL)
