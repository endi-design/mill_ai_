"""Compare every difficulty from difficulty_config.py against a Random player.

Run from the project folder:
    python difficulty_vs_random.py --games-per-seat 25 --output-dir results/difficulty_vs_random

Only the MCTS iteration budget varies between difficulties. Every MCTS agent
uses the same DEFAULT_C and rollout policy. Random chooses legal actions uniformly.
Each matchup alternates MCTS between player 1 and player 2.

Outputs (no third-party plotting libraries required):
    difficulty_vs_random_games.csv
    difficulty_vs_random_summary.csv
    difficulty_vs_random_score.svg
    difficulty_vs_random_time.svg

Requires famnit_gym at runtime, plus your existing run.py, expansion_practice.py
and difficulty_config.py in this directory.
"""

from __future__ import annotations

import argparse
import csv
import random
from html import escape
from pathlib import Path
from time import perf_counter

from difficulty_config import DEFAULT_C, LEVEL_ITERATIONS
from run import remaining_moves_from_info, verify_move


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_svg(rows: list[dict], key: str, title: str, ylabel: str, path: Path,
             maximum: float | None = None):
    """Write a small, dependency-free SVG bar chart for the poster."""
    width = max(640, 160 + 130 * len(rows))
    height = 365
    left, right, top, bottom = 92, 32, 70, 88
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(r[key]) for r in rows]
    upper = maximum if maximum is not None else max(1.0, max(values) * 1.15)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="19" font-weight="bold">{escape(title)}</text>',
             f'<text transform="translate(23 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12">{escape(ylabel)}</text>']
    for tick in range(5):
        value = upper * tick / 4
        y = top + plot_h * (1 - tick / 4)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>')
    stride = plot_w / len(rows)
    for i, (r, v) in enumerate(zip(rows, values)):
        bar_w = min(74.0, stride * .55)
        x = left + stride * (i + .5) - bar_w / 2
        h = max(0, min(plot_h, v / upper * plot_h))
        y = top + plot_h - h
        label = f'{r["difficulty"]} ({r["iterations"]})'
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#3779ac"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{max(top+12, y-7):.1f}" text-anchor="middle" font-family="Arial" font-size="12">{v:.1f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+24:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{escape(label)}</text>')
    parts.append(f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="12">Difficulty (MCTS iterations)</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def run_experiment(games_per_seat: int, seed: int, output_dir: Path):
    """Run each configured MCTS difficulty against a uniform Random opponent."""
    from famnit_gym.envs import mill
    from expansion_practice import choose_mcts_move

    if games_per_seat < 1:
        raise ValueError("--games-per-seat must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    game_rows = []
    summary_rows = []
    env = mill.env(render_mode=None)
    try:
        for level_index, (difficulty, iterations) in enumerate(LEVEL_ITERATIONS.items()):
            wins = losses = draws = decisions = 0
            total_seconds = 0.0
            print(f"\n{difficulty}: {iterations} iterations vs Random, C={DEFAULT_C}")
            for i in range(games_per_seat * 2):
                random.seed(seed + level_index * 100_000 + i)
                env.reset()
                mcts_seat = 1 if i % 2 == 0 else 2
                winner = "draw"
                game_seconds = 0.0
                game_decisions = 0
                plies = 0
                while True:
                    _, _, terminated, truncated, info = env.last()
                    if terminated:
                        loser = 1 if info["agent"] == "player_1" else 2
                        winner = "mcts" if 3 - loser == mcts_seat else "random"
                        break
                    if truncated:
                        break

                    player = 1 if info["agent"] == "player_1" else 2
                    if player == mcts_seat:
                        model = mill.transition_model(env.unwrapped)
                        remaining = remaining_moves_from_info(info, player)
                        start = perf_counter()
                        move = choose_mcts_move(model, player, iterations, DEFAULT_C, remaining)
                        seconds = perf_counter() - start
                        decisions += 1
                        game_decisions += 1
                        game_seconds += seconds
                        total_seconds += seconds
                    else:
                        move = random.choice(info["legal_moves"])
                    env.step(verify_move(move, info["legal_moves"]))
                    plies += 1

                if winner == "mcts":
                    wins += 1
                elif winner == "random":
                    losses += 1
                else:
                    draws += 1
                game_rows.append({
                    "difficulty": difficulty, "iterations": iterations, "C": DEFAULT_C,
                    "game": i + 1, "mcts_seat": mcts_seat, "winner": winner,
                    "plies": plies, "mcts_decisions": game_decisions,
                    "mcts_seconds": round(game_seconds, 6),
                })
                print(f"  Game {i+1}/{games_per_seat * 2}: {winner}")

            games = games_per_seat * 2
            result = {
                "difficulty": difficulty, "iterations": iterations, "C": DEFAULT_C,
                "games": games, "mcts_wins": wins, "random_wins": losses,
                "draws": draws, "mcts_win_rate_percent": round(100 * wins / games, 2),
                "mcts_score_rate_percent": round(100 * (wins + .5 * draws) / games, 2),
                "average_ms_per_mcts_decision": round(1000 * total_seconds / decisions, 3) if decisions else 0.0,
                "mcts_decisions": decisions,
            }
            summary_rows.append(result)
            print(f"  RESULT: {wins} wins, {losses} losses, {draws} draws; "
                  f"score={result['mcts_score_rate_percent']}%; "
                  f"time={result['average_ms_per_mcts_decision']} ms/decision")
    finally:
        env.close()

    write_csv(output_dir / 'difficulty_vs_random_games.csv', game_rows)
    write_csv(output_dir / 'difficulty_vs_random_summary.csv', summary_rows)
    draw_svg(summary_rows, 'mcts_score_rate_percent', 'MCTS difficulties vs Random',
             'Score rate (%)', output_dir / 'difficulty_vs_random_score.svg', maximum=100.0)
    draw_svg(summary_rows, 'average_ms_per_mcts_decision', 'MCTS computation time vs Random',
             'Average ms / MCTS decision', output_dir / 'difficulty_vs_random_time.svg')
    print(f"\nSaved CSV files and SVG graphs to {output_dir}")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-per-seat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--output-dir', type=Path, default=Path('results/difficulty_vs_random'))
    args = parser.parse_args()
    run_experiment(args.games_per_seat, args.seed, args.output_dir)


if __name__ == '__main__':
    main()
