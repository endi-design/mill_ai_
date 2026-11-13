"""Random player and three difficulty presets (standard library only)."""

from __future__ import annotations

import math
import random
from typing import Any

from mcts import MCTSAgent, normalize_moves


class RandomAgent:
    """Baseline: choose any legal action with equal probability.

    This is NOT intelligent, but it gives us a control group when measuring
    the strength of MCTS. Each agent has its own reproducible random stream.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, model: Any, player: int, remaining_plies: int = 200) -> list[int]:
        legal = normalize_moves(model.legal_moves(player))
        if not legal:
            raise ValueError("RandomAgent cannot choose from zero legal moves")
        return list(self.rng.choice(legal))


# Difficulties change how long the AI thinks and optionally how often it
# commits a deliberate mistake. Tunable presets, not measured guarantees.
DIFFICULTY_PRESETS = {
    "easy": {"iterations": 30, "mistake_rate": 0.12},
    "medium": {"iterations": 120, "mistake_rate": 0.02},
    "hard": {"iterations": 400, "mistake_rate": 0.0},
}


def make_agent(
    name: str,
    seed: int | None = None,
    iterations: int = 150,
    exploration: float = math.sqrt(2),
):
    """Build any player using one common factory.

    Name may be: random, mcts, easy, medium, hard.
    Using the same choose_move() interface makes it possible to match ANY
    pair of player types without rewriting the game loop.
    """
    if name == "random":
        return RandomAgent(seed)
    if name == "mcts":
        return MCTSAgent(iterations=iterations, exploration=exploration, seed=seed)
    if name in DIFFICULTY_PRESETS:
        settings = DIFFICULTY_PRESETS[name]
        return MCTSAgent(
            iterations=settings["iterations"],
            exploration=exploration,
            mistake_rate=settings["mistake_rate"],
            seed=seed,
        )
    raise ValueError(f"Unknown agent type: {name!r}")
