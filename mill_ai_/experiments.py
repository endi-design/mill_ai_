"""Reproducible seminar experiments + CSV and SVG plots (stdlib only).

We do not import pandas or matplotlib: the assignment permits only FAMNIT
Gym plus Python's standard library. SVG is a normal vector-image format;
we create the SVG text ourselves to comply with that requirement.

Studies:
  uct-random       : each UCT C value vs random
  uct-mcts         : each UCT C value vs a reference MCTS agent
  difficulties     : easy/medium/hard in round-robin matches
"""

from __future__ import annotations

import csv
from pathlib import Path
from html import escape
from statistics import mean

from game import play_match

DEFAULT_C_VALUES = (0.0, 0.5, 1.0, 1.41421356237, 2.0)


def save_csv(path: Path, rows: list[dict]) -> None:
    """CSV can be opened in LibreOffice/Excel or used for a poster."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bars(path: Path, rows: list[dict], column: str, title: str,
              y_label: str, maximum: float | None = None) -> None:
    """Write a readable static bar chart in SVG using no third-party library."""
    if not rows:
        return
    width, height = max(740, 105 * len(rows) + 140), 470
    left, right, top, bottom = 82, 35, 65, 115
    chart_w = width - left - right
    chart_h = height - top - bottom
    values = [float(row[column]) for row in rows]
    max_y = maximum if maximum is not None else max(0.01, max(values) * 1.1)
    step = chart_w / len(rows)
    bar_w = min(72, step * 0.68)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="31" font-family="Arial" text-anchor="middle" font-size="20" fill="#1b2936">{escape(title)}</text>',
        f'<text x="18" y="{top+chart_h/2:.1f}" transform="rotate(-90,18,{top+chart_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">{escape(y_label)}</text>',
    ]
    for i in range(6):
        y_value = max_y * i / 5
        y = top + chart_h * (1 - i / 5)
        parts.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e3e9ec"/>',
            f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{y_value:.2f}</text>',
        ])
    for i, row in enumerate(rows):
        value = values[i]
        center_x = left + step * (i + 0.5)
        bar_h = chart_h * value / max_y
        y = top + chart_h - bar_h
        parts.extend([
            f'<rect x="{center_x-bar_w/2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#326ba8" rx="3"/>',
            f'<text x="{center_x:.2f}" y="{y-8:.2f}" font-family="Arial" font-size="11" text-anchor="middle" fill="#1f2937">{value:.2f}</text>',
            f'<text x="{center_x:.2f}" y="{top+chart_h+22}" font-family="Arial" font-size="12" text-anchor="middle" fill="#1f2937">{escape(str(row["label"]))}</text>',
        ])
    parts.append(f'<text x="{width/2:.1f}" y="{height-28}" font-family="Arial" font-size="12" text-anchor="middle" fill="#586574">Results depend on game seeds and sample size. Do not claim statistical certainty from few games.</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def run_experiments(study: str, games_per_seat: int = 1,
                    iterations: int = 60, baseline_iterations: int = 60,
                    seed: int = 2026,
                    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
                    output_dir: str = "results") -> None:
    """Run balanced matches and save source data plus charts.

    Each comparison plays equally many games with the tested AI as player 1
    and player 2. This controls some first-move advantage. Using seeds makes
    experiments reproducible (elapsed timings still vary by computer).
    """
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be >= 1")
    if iterations < 1 or baseline_iterations < 1:
        raise ValueError("iteration counts must be >= 1")
    if study not in ("uct-random", "uct-mcts", "difficulties"):
        raise ValueError("Unknown experiment type")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    all_games: list[dict] = []
    summary: list[dict] = []

    if study == "difficulties":
        comparisons = [
            ("easy", "medium", None),
            ("easy", "hard", None),
            ("medium", "hard", None),
        ]
    else:
        reference = "random" if study == "uct-random" else "mcts"
        comparisons = [
            ("mcts", reference, c) for c in c_values
        ]

    for group_number, (tested, opponent, c) in enumerate(comparisons):
        label = (f"C={c:g}" if c is not None else f"{tested} vs {opponent}")
        wins, draws, losses = 0, 0, 0
        times: list[float] = []
        turns: list[int] = []
        move_times: list[float] = []

        # Test EACH configuration with each seat, not just as first player.
        for seat in (1, 2):
            for repeat in range(games_per_seat):
                current_seed = seed + group_number * 10000 + repeat
                first = tested if seat == 1 else opponent
                second = opponent if seat == 1 else tested

                # When compared against MCTS, the reference uses sqrt(2).
                # In difficulty studies, each preset has its fixed settings.
                if study == "difficulties":
                    first_c = second_c = 2 ** 0.5
                else:
                    first_c = c if seat == 1 else 2 ** 0.5
                    second_c = c if seat == 2 else 2 ** 0.5

                result = play_match(
                    player1=first, player2=second,
                    iterations1=iterations if first == "mcts" else baseline_iterations,
                    iterations2=iterations if second == "mcts" else baseline_iterations,
                    exploration1=first_c, exploration2=second_c,
                    seed=current_seed, verbose=False,
                )

                if result.winner == seat:
                    outcome = "win"
                    wins += 1
                elif result.winner == 0:
                    outcome = "draw"
                    draws += 1
                else:
                    outcome = "loss"
                    losses += 1

                think_seconds = (result.seconds_player1 if seat == 1
                                 else result.seconds_player2)
                tested_moves = (result.moves_player1 if seat == 1
                                else result.moves_player2)
                times.append(think_seconds)
                turns.append(result.total_turns)
                move_times.append(think_seconds / tested_moves if tested_moves else 0)
                all_games.append({
                    "study": study, "label": label, "tested": tested,
                    "opponent": opponent, "c": "" if c is None else round(c, 6),
                    "tested_seat": seat, "repeat": repeat + 1,
                    "seed": current_seed, "iterations": iterations,
                    "winner": result.winner, "outcome": outcome,
                    "reason": result.reason, "total_turns": result.total_turns,
                    "tested_move_count": tested_moves,
                    "tested_total_think_seconds": round(think_seconds, 6),
                })

        total_games = wins + draws + losses
        scored = (wins + 0.5 * draws) / total_games
        row = {
            "study": study, "label": label, "tested": tested, "opponent": opponent,
            "c": "" if c is None else round(c, 6),
            "games": total_games, "wins": wins, "draws": draws,
            "losses": losses, "win_rate": round(wins / total_games, 4),
            "score_rate": round(scored, 4),
            "average_match_turns": round(mean(turns), 2),
            "average_think_sec_per_game": round(mean(times), 4),
            "average_think_sec_per_move": round(mean(move_times), 5),
        }
        summary.append(row)
        print(f"{label:<25} W={wins:>2} D={draws:>2} L={losses:>2} "
              f"score={100*scored:5.1f}% | think/move={row['average_think_sec_per_move']:.4f}s")

    # Raw game logs are saved so conclusions are auditable and reproducible.
    save_csv(target / "games.csv", all_games)
    save_csv(target / "summary.csv", summary)
    plot_bars(target / "score_rate.svg", summary, "score_rate",
              f"{study}: score rate", "Score: win=1, draw=0.5, loss=0", maximum=1.0)
    plot_bars(target / "time_per_move.svg", summary, "average_think_sec_per_move",
              f"{study}: mean thinking time", "Seconds / agent move")
    print(f"\nSaved results to: {target.resolve()}")
