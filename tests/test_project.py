"""Offline unit tests for helpers and CLI options (no FAMNIT required)."""

import contextlib
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import run


class TestProject(unittest.TestCase):
    """Check CLI arguments, simple counters and illegal move rejection."""

    def test_remaining_moves(self):
        self.assertEqual(run.remaining_moves_from_info({"move": 1}, 1), 200)
        self.assertEqual(run.remaining_moves_from_info({"move": 1}, 2), 199)
        self.assertEqual(run.remaining_moves_from_info({"move": 100}, 1), 2)
        self.assertEqual(run.remaining_moves_from_info({"move": 100}, 2), 1)

    def test_move_legality(self):
        self.assertEqual(run.verify_move([0, 2, 0], [[0, 2, 0]]), [0, 2, 0])
        with self.assertRaises(ValueError):
            run.verify_move([9, 9, 9], [[0, 2, 0]])

    def test_difficulty_mapping(self):
        self.assertEqual(run.LEVEL_ITERATIONS, {"easy": 1000, "medium": 500, "hard": 1000})

    def test_command_line_parsing(self):
        parser = run.build_parser()
        args = parser.parse_args(["match", "--p1", "mcts", "--p2", "mcts",
                                  "--c1", "0.5", "--c2", "2.0", "--games-per-seat", "4"])
        self.assertEqual(args.c1, 0.5)
        self.assertEqual(args.c2, 2.0)
        self.assertEqual(args.games_per_seat, 4)
        exp = parser.parse_args(["experiment", "--study", "time-budget",
                                 "--games-per-seat", "5"])
        self.assertEqual(exp.study, "time-budget")
        human = parser.parse_args(["human", "--difficulty", "hard", "--human-seat", "2"])
        self.assertEqual(human.human_seat, 2)

    def test_uct_and_rollout_with_stub_model(self):
        """Exercise the supplied MCTS code without claiming to test real Mill rules."""
        import types
        mock_pkg = types.ModuleType("famnit_gym")
        mock_envs = types.ModuleType("famnit_gym.envs")
        mock_mill = types.ModuleType("famnit_gym.envs.mill")
        mock_envs.mill = mock_mill
        with patch.dict(sys.modules, {
            "famnit_gym": mock_pkg,
            "famnit_gym.envs": mock_envs,
            "famnit_gym.envs.mill": mock_mill,
        }):
            import expansion_practice
        from simulate import simulate_random_game

        class FakeModel:
            def __init__(self, steps=0):
                self.steps = steps
            def clone(self):
                return FakeModel(self.steps)
            def game_over(self):
                return self.steps >= 4
            def legal_moves(self, player):
                return [[0, 1, 0], [0, 2, 0]] if self.steps < 4 else []
            def make_move(self, player, move):
                self.steps += 1
            def get_phase(self, player):
                return "lost" if player == 1 and self.game_over() else "moving"

        model = FakeModel()
        self.assertEqual(simulate_random_game(model, 1, 4), 2)
        action = expansion_practice.choose_mcts_move(model, 1, 10, 1.4, 4)
        self.assertIn(action, model.legal_moves(1))
        self.assertEqual(model.steps, 0, "Simulations must not change the original board")


if __name__ == "__main__":
    unittest.main()
