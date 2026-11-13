"""SMOKE TEST: Run this AFTER installing FAMNIT Gym on your computer.

Unlike tests/test_mcts.py, this loads the real Mill game and checks that
our search agent returns an action legal in the actual environment.
"""

from famnit_gym.envs import mill

from mcts import MCTSAgent, normalize_moves
from game import play_match


def main():
    env = mill.env(render_mode="ansi")
    try:
        env.reset()
        _, _, _, _, info = env.last()
        model = mill.transition_model(env.unwrapped)
        before = list(model.get_state())

        agent = MCTSAgent(iterations=12, seed=9)
        action = agent.choose_move(model, player=1)
        legal = normalize_moves(info["legal_moves"])
        assert tuple(action) in legal, "MCTS returned an illegal Mill action"
        assert list(model.get_state()) == before, "MCTS modified the real model"
        print("PASS: real Mill environment accepts MCTS action", action)
    finally:
        env.close()

    # One complete integration game at LOW iterations (faster than defaults).
    result = play_match("mcts", "random", iterations1=8,
                        seed=987, verbose=False)
    assert result.winner in (0, 1, 2)
    print("PASS: completed match with winner:", result.winner,
          "turns:", result.total_turns)


if __name__ == "__main__":
    main()
