"""Unit tests with a tiny game model; no third-party testing library needed.

This intentionally tests the algorithm independently of FAMNIT Gym. The
actual FAMNIT integration is checked by smoke_test.py on the student's PC.
Run from the project folder: python -m unittest discover -s tests -v
"""

import math
import unittest

from agents import RandomAgent, make_agent
from mcts import MCTSAgent, Node, normalize_moves


class TakeAway:
    """A two-player toy game that implements the same API as Mill.

    There are 'stones' stones; each turn a player removes 1 or 2 stones.
    The player removing the last stone wins. Moves are formatted as Mill
    moves [remove_count, 0, 0] solely so this tests the agent's interface.
    """

    def __init__(self, stones=4, last_mover=2):
        self.stones = stones
        self.last_mover = last_mover

    def clone(self):
        return TakeAway(self.stones, self.last_mover)

    def legal_moves(self, player):
        if self.game_over():
            return []
        return [[n, 0, 0] for n in (1, 2) if n <= self.stones]

    def make_move(self, player, move):
        amount = int(move[0])
        if amount not in (1, 2) or amount > self.stones:
            raise ValueError("illegal toy-game move")
        self.stones -= amount
        self.last_mover = player

    def game_over(self):
        return self.stones == 0

    def get_phase(self, player):
        if self.game_over() and player != self.last_mover:
            return "lost"
        return "moving"


class TestMCTS(unittest.TestCase):
    def test_normalize_actions(self):
        self.assertEqual(normalize_moves([[0, 1, 0], [2, 3, 4]]),
                         [(0, 1, 0), (2, 3, 4)])

    def test_single_legal_move(self):
        agent = MCTSAgent(iterations=5, seed=1)
        state = TakeAway(stones=1)
        self.assertEqual(agent.choose_move(state, 1), [1, 0, 0])
        self.assertEqual(state.stones, 1)  # Never modify caller's board.

    def test_find_forced_immediate_win(self):
        # From 2 stones, taking both immediately wins. With enough search,
        # adversarial MCTS should strongly prefer that move.
        agent = MCTSAgent(iterations=350, exploration=1.0, seed=7)
        state = TakeAway(stones=2)
        self.assertEqual(agent.choose_move(state, 1), [2, 0, 0])
        self.assertEqual(state.stones, 2)

    def test_random_agent_returns_legal_move(self):
        state = TakeAway(stones=2)
        agent = RandomAgent(seed=3)
        self.assertIn(agent.choose_move(state, 1), state.legal_moves(1))

    def test_uct_components(self):
        root = Node(state=TakeAway(stones=5), to_move=1,
                    just_moved=2, depth=0, ply_limit=10)
        root.visits = 100
        child = Node(state=TakeAway(stones=4), to_move=2,
                     just_moved=1, depth=1, ply_limit=10, parent=root)
        child.visits = 10
        child.reward_sum = 4.0
        expected = 0.4 + 1.4 * math.sqrt(math.log(100) / 10)
        self.assertAlmostEqual(child.uct_score(1.4), expected)

    def test_short_rollout_safely_draws_at_limit(self):
        agent = MCTSAgent(iterations=7, seed=2)
        state = TakeAway(stones=20)
        action = agent.choose_move(state, 1, remaining_plies=1)
        self.assertIn(action, state.legal_moves(1))
        self.assertEqual(state.stones, 20)

    def test_difficulty_levels_exist(self):
        for name in ("easy", "medium", "hard"):
            self.assertIsInstance(make_agent(name, seed=1), MCTSAgent)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            MCTSAgent(iterations=0)
        with self.assertRaises(ValueError):
            MCTSAgent(exploration=-0.1)


if __name__ == "__main__":
    unittest.main()
