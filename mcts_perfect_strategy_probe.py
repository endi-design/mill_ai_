"""Research exercise: repeat MCTS searches from the starting board using
different seeds and budgets. Finds promising openings; never proves perfect play.

Notes: This file is part of the FAMNIT Mill seminar project.
"""


import csv
import os
import random
from collections import Counter
from pathlib import Path

from famnit_gym.envs import mill
from expansion_practice import mcts_iteration
from difficulty_config import DEFAULT_C

# Start small; increase the largest budget later if desired.
BUDGETS = [100, 1000, 5000]
SEEDS = [11, 29, 47]
C = DEFAULT_C
TOP_MOVES = 5


def create_root(model, player):
    """Make a root node using the same fields as expansion_practice.py."""
    return {
        "state": model.clone(),
        "to_move": player,
        "just_moved": 3 - player,
        "move": None,
        "parent": None,
        "depth": 0,
        "visits": 0,
        "reward_sum": 0,
        "children": [],
        "untried_moves": model.legal_moves(player),
    }


def move_text(move):
    """Format a three-number move as readable text for the report."""
    return str([int(x) for x in move])


def evaluate_root(root):
    """Rank moves by visits, as choose_best_move() does.

    Signed average outcome is +1 for a win and -1 for a loss from the
    perspective of the player who made the root move; draws score 0.
    It is NOT a probability and is NOT a proven game-theoretic value.
    """
    ranked = sorted(
        root["children"],
        key=lambda child: child["visits"],
        reverse=True,
    )
    return ranked


def run():
    """Repeat opening-search probes at different budgets and seeds, compare
    most-visited choices, and write the observations to CSV."""
    out_dir = Path(os.getenv("MILL_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mcts_opening_probe.csv"
    rows = []
    choices = {budget: [] for budget in BUDGETS}
    game = mill.env(render_mode=None)

    try:
        for seed in SEEDS:
            random.seed(seed)
            game.reset()
            model = mill.transition_model(game.unwrapped)
            player = 1
            remaining_moves = 200  # Opening: 100 rounds x 2 individual moves
            root = create_root(model, player)
            previous_budget = 0

            print("\n" + "=" * 45)
            print("Random seed:", seed)
            print("=" * 45)

            for budget in BUDGETS:
                if budget <= previous_budget:
                    raise ValueError("BUDGETS must be strictly increasing")

                # Continue the SAME tree, rather than starting over.
                for _ in range(budget - previous_budget):
                    mcts_iteration(root, C, remaining_moves)
                previous_budget = budget

                ranked = evaluate_root(root)
                if not ranked:
                    raise RuntimeError("No moves explored. Check MCTS code.")

                best_move = move_text(ranked[0]["move"])
                choices[budget].append(best_move)

                print("\nIterations:", budget)
                print("Current most-visited opening:", best_move)
                print("Top", min(TOP_MOVES, len(ranked)), "moves:")

                for rank, child in enumerate(ranked, start=1):
                    mean_score = child["reward_sum"] / child["visits"]
                    if rank <= TOP_MOVES:
                        print(
                            f"  {rank}. {move_text(child['move'])} "
                            f"visits={child['visits']} "
                            f"mean_signed_outcome={mean_score:.3f}"
                        )

                    rows.append({
                        "seed": seed,
                        "iterations": budget,
                        "rank": rank,
                        "move": move_text(child["move"]),
                        "visits": child["visits"],
                        "mean_signed_outcome": round(mean_score, 5),
                        "most_visited": rank == 1,
                    })

        print("\n" + "=" * 45)
        print("STABILITY ACROSS RANDOM SEEDS")
        print("=" * 45)
        for budget in BUDGETS:
            counts = Counter(choices[budget])
            print(f"{budget} iterations: {dict(counts)}")
            print(
                "Most common candidate:", counts.most_common(1)[0][0],
                "(selected in", counts.most_common(1)[0][1],
                "out of", len(SEEDS), "runs)"
            )

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "iterations", "rank", "move", "visits",
                "mean_signed_outcome", "most_visited",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print("\nSaved:", out_path)
        print("\nIMPORTANT: This discovers CANDIDATE openings, not a perfect-play proof.")
        print("A proof requires evaluating all relevant opponent responses")
        print("through terminal states, or a certified game-solving database.")

    finally:
        game.close()


if __name__ == "__main__":
    run()
