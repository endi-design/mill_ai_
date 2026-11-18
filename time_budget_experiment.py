"""Nine Men's Morris: wall-clock MCTS time budgets experiment.

Requires famnit_gym and your existing expansion_practice.py next to this file.
Run: python time_budget_experiment.py
Change GAMES_PER_MATCHUP to 50 for a larger tournament (150 total games).
"""

import csv
import random
from pathlib import Path
from time import perf_counter
import os
from itertools import combinations

from famnit_gym.envs import mill
from expansion_practice import mcts_iteration, choose_best_move
from difficulty_config import DIFFICULTIES, TIME_BUDGETS, DEFAULT_C as FIXED_C

# Three head-to-head pairings; games alternate which agent moves first.
GAMES_PER_MATCHUP = int(os.getenv("MILL_GAMES_PER_MATCHUP", "10"))
MATCHUPS = list(combinations(DIFFICULTIES, 2))

RANDOM_SEED = int(os.getenv("MILL_SEED", "20261117"))
METHOD = "time_budget"
# Seconds of thinking per decision. Actual time may overshoot because
# a currently running rollout cannot be interrupted mid-iteration.


def get_remaining_moves(info, player):
    """Remaining individual moves under the 100-round match limit."""
    round_number = max(1, min(100, int(info["move"])))
    remaining = 2 * (101 - round_number)
    if player == 2:
        remaining -= 1
    return max(0, remaining)


def select_move(model, player, difficulty, remaining_moves):
    """Stop searching when the decision's wall-clock budget expires."""
    legal_moves = model.legal_moves(player)
    if not legal_moves:
        raise ValueError("No legal move: should be a terminal position")
    if len(legal_moves) == 1:
        return legal_moves[0], 0, False

    # Same root structure as your choose_mcts_move() implementation.
    root = {
        "state": model.clone(),
        "to_move": player,
        "just_moved": 3 - player,
        "move": None,
        "parent": None,
        "depth": 0,
        "visits": 0,
        "reward_sum": 0,
        "children": [],
        "untried_moves": legal_moves,
    }

    deadline = perf_counter() + TIME_BUDGETS[difficulty]
    iterations_done = 0

    # Always run at least one iteration. Check the clock between full
    # iterations; stopping inside an iteration would leave a partial update.
    while iterations_done == 0 or perf_counter() < deadline:
        mcts_iteration(root, FIXED_C, remaining_moves)
        iterations_done += 1

    best_move = choose_best_move(root)
    if best_move is None:
        raise RuntimeError("MCTS created no root child")
    return best_move, iterations_done, False

def main():
    random.seed(RANDOM_SEED)  # Controls Python random choices/rollouts.
    # If FAMNIT uses separate random generators, seed those separately
    # for fully reproducible results.

    levels = list(DIFFICULTIES)
    wins = {level: 0 for level in levels}
    draws_by_level = {level: 0 for level in levels}
    games_by_level = {level: 0 for level in levels}
    total_time = {level: 0.0 for level in levels}
    decisions = {level: 0 for level in levels}
    random_decisions = {level: 0 for level in levels}
    search_iterations = {level: 0 for level in levels}
    game_rows = []
    matchup_rows = []

    game = mill.env(render_mode=None)
    try:
        for level_a, level_b in MATCHUPS:
            pair_wins = {level_a: 0, level_b: 0}
            pair_draws = 0
            print("\n===", METHOD, ":", level_a, "vs", level_b, "===")

            for game_number in range(1, GAMES_PER_MATCHUP + 1):
                game.reset()
                if game_number % 2 == 1:
                    seats = {1: level_a, 2: level_b}
                else:
                    seats = {1: level_b, 2: level_a}

                winner = "draw"
                plies = 0
                move_times = {1: 0.0, 2: 0.0}

                while True:
                    _, _, terminated, truncated, info = game.last()

                    if terminated:
                        # Matches the terminal-agent convention from the
                        # original tournament.py used in this project.
                        losing_player = 1 if info["agent"] == "player_1" else 2
                        winner = seats[3 - losing_player]
                        wins[winner] += 1
                        pair_wins[winner] += 1
                        break

                    if truncated:
                        pair_draws += 1
                        draws_by_level[level_a] += 1
                        draws_by_level[level_b] += 1
                        break

                    player = 1 if info["agent"] == "player_1" else 2
                    level = seats[player]
                    model = mill.transition_model(game.unwrapped)
                    remaining = get_remaining_moves(info, player)

                    start = perf_counter()
                    move, iterations_used, used_random = select_move(
                        model, player, level, remaining
                    )
                    elapsed = perf_counter() - start

                    total_time[level] += elapsed
                    decisions[level] += 1
                    search_iterations[level] += iterations_used
                    if used_random:
                        random_decisions[level] += 1
                    move_times[player] += elapsed

                    # Never let the environment silently substitute an
                    # invalid AI move with a random legal move.
                    chosen = [int(value) for value in move]
                    legal = [[int(value) for value in m]
                             for m in info["legal_moves"]]
                    if chosen not in legal:
                        raise ValueError(
                            f"Illegal move {chosen} for {level}; legal={legal}"
                        )
                    game.step(chosen)
                    plies += 1

                games_by_level[level_a] += 1
                games_by_level[level_b] += 1
                game_rows.append({
                    "method": METHOD,
                    "matchup": f"{level_a}_vs_{level_b}",
                    "game": game_number,
                    "player1": seats[1],
                    "player2": seats[2],
                    "winner": winner,
                    "plies": plies,
                    "player1_seconds": round(move_times[1], 6),
                    "player2_seconds": round(move_times[2], 6),
                })
                print(f"Game {game_number}: {winner}")

            matchup_rows.append({
                "method": METHOD,
                "level_a": level_a,
                "level_b": level_b,
                "level_a_wins": pair_wins[level_a],
                "level_b_wins": pair_wins[level_b],
                "draws": pair_draws,
                "games": GAMES_PER_MATCHUP,
            })
            print("Pair results:", pair_wins, "; draws:", pair_draws)
    finally:
        game.close()

    summary_rows = []
    print("\n=== FINAL RESULTS:", METHOD, "===")
    for level in levels:
        n_games = games_by_level[level]
        n_decisions = decisions[level]
        avg_seconds = total_time[level] / n_decisions if n_decisions else 0
        avg_iterations = (search_iterations[level] / n_decisions
                          if n_decisions else 0)
        win_rate = 100 * wins[level] / n_games if n_games else 0
        print(
            f"{level}: {wins[level]} wins / {n_games} games; "
            f"{win_rate:.1f}% wins; {avg_seconds:.4f} s/decision; "
            f"{avg_iterations:.1f} MCTS iterations/decision"
        )
        summary_rows.append({
            "method": METHOD,
            "difficulty": level,
            "setting": f"time_budget_seconds={TIME_BUDGETS[level]}",
            "games": n_games,
            "wins": wins[level],
            "losses": n_games - wins[level] - draws_by_level[level],
            "draws": draws_by_level[level],
            "win_rate_percent": round(win_rate, 2),
            "decisions": n_decisions,
            "random_decisions": random_decisions[level],
            "total_search_iterations": search_iterations[level],
            "average_iterations_per_decision": round(avg_iterations, 2),
            "average_seconds_per_decision": round(avg_seconds, 6),
        })

    folder = Path(os.getenv("MILL_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (
        ("games", game_rows),
        ("matchups", matchup_rows),
        ("summary", summary_rows),
    ):
        path = folder / f"{METHOD}_{suffix}.csv"
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", path.name)

    # Optional poster-ready graph. No dependency on pandas.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed: CSV results were still saved.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(levels, [r["win_rate_percent"] for r in summary_rows])
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("Win rate (%)")
        axes[0].set_title(f"{METHOD}: game outcomes")
        axes[1].bar(levels, [1000*r["average_seconds_per_decision"]
                             for r in summary_rows])
        axes[1].set_ylabel("Average thinking time (ms)")
        axes[1].set_title(f"{METHOD}: computation time")
        fig.tight_layout()
        graph = folder / f"{METHOD}_comparison.png"
        fig.savefig(graph, dpi=160)
        plt.close(fig)
        print("Saved:", graph.name)


if __name__ == "__main__":
    main()
