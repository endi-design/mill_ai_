"""Optional UCT-vs-Random comparison used by `run.py experiment --study uct-random`.

This is deliberately separate from the C-vs-C experiment: each exploration
constant is tested against Random in equally many games with swapped seats.
"""

import csv
import random
from pathlib import Path
from time import perf_counter

from run import remaining_moves_from_info, verify_move
from difficulty_config import C_STUDY_VALUES as CONSTANTS


def run_uct_vs_random(games_per_seat, iterations, seed, output_dir):
    """Compare C values against Random using twice games_per_seat matches each."""
    from famnit_gym.envs import mill
    from expansion_practice import choose_mcts_move
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = []
    env = mill.env(render_mode=None)
    try:
        for level, C in CONSTANTS.items():
            wins, losses, draws, search_time, decisions = 0, 0, 0, 0.0, 0
            for index in range(2 * games_per_seat):
                env.reset()
                mcts_seat = 1 if index % 2 == 0 else 2
                winner = "draw"
                while True:
                    _, _, terminated, truncated, info = env.last()
                    if terminated:
                        loser = 1 if info["agent"] == "player_1" else 2
                        winner = level if 3 - loser == mcts_seat else "random"
                        if winner == level:
                            wins += 1
                        else:
                            losses += 1
                        break
                    if truncated:
                        draws += 1
                        break
                    player = 1 if info["agent"] == "player_1" else 2
                    model = mill.transition_model(env.unwrapped)
                    if player == mcts_seat:
                        start = perf_counter()
                        move = choose_mcts_move(model, player, iterations, C,
                                                remaining_moves_from_info(info, player))
                        search_time += perf_counter() - start
                        decisions += 1
                    else:
                        move = random.choice(info["legal_moves"])
                    env.step(verify_move(move, info["legal_moves"]))
                rows.append({"level": level, "C": C, "game": index + 1,
                             "mcts_seat": mcts_seat, "winner": winner})
                print(f"C={C}, game {index + 1}/{2 * games_per_seat}: {winner}")
            summaries.append({"level": level, "C": C,
                              "mcts_iterations": iterations,
                              "games": 2 * games_per_seat, "wins": wins,
                              "losses": losses, "draws": draws,
                              "win_rate_percent": round(100 * wins / (2 * games_per_seat), 2),
                              "average_seconds_per_decision": round(search_time / decisions, 6) if decisions else 0.0})
    finally:
        env.close()
    for filename, data in (("uct_random_games.csv", rows),
                           ("uct_random_summary.csv", summaries)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        print("Saved:", output_dir / filename)
    print("Summary:")
    for row in summaries:
        print(f"  C={row['C']}: {row['wins']}/{row['games']} wins, "
              f"{row['average_seconds_per_decision']:.4f}s/decision")
