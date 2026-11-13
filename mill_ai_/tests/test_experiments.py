"""Check experiment output without running hundreds of expensive Mill games."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_experiments
from game import MatchResult


class TestExperimentOutputs(unittest.TestCase):
    def test_writes_csv_and_svg_without_third_party_plotting(self):
        # Replace a whole match with a deterministic fake result. This tests
        # CSV/SVG output logic, NOT agent skill or real Mill integration.
        fake = MatchResult(winner=1, reason="test-only", total_turns=12,
                           seconds_player1=0.25, seconds_player2=0.02,
                           moves_player1=6, moves_player2=6)
        with tempfile.TemporaryDirectory() as temp:
            with patch("experiments.play_match", return_value=fake):
                run_experiments(
                    "uct-random", games_per_seat=1,
                    iterations=2, c_values=(0.0, 1.0), output_dir=temp
                )
            for name in ("games.csv", "summary.csv", "score_rate.svg",
                         "time_per_move.svg"):
                path = Path(temp) / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0)
            self.assertIn("<svg", (Path(temp) / "score_rate.svg").read_text())


if __name__ == "__main__":
    unittest.main()
