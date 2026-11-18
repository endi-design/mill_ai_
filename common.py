"""Shared helpers for controlled MCTS experiments using the real FAMNIT Mill game.

This file never changes board rules: the installed FAMNIT engine supplies all
legal moves, including moves that complete diagonal mills.
"""
from __future__ import annotations

import csv
from pathlib import Path
import random
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def remaining_moves(info: dict, player: int) -> int:
    """How many individual turns remain before FAMNIT's 100-round limit."""
    round_number = min(100, max(1, int(info["move"])))
    return max(0, 2 * (101 - round_number) - (1 if player == 2 else 0))


def normalise(move):
    """Convert NumPy/list move values to normal Python integer lists."""
    return [int(x) for x in move]


def check_move(move, raw_legal):
    """Reject illegal actions instead of letting the engine replace them."""
    selected = normalise(move)
    legal = [normalise(action) for action in raw_legal]
    if selected not in legal:
        raise ValueError(f"Illegal move {selected}; legal moves: {legal}")
    return selected


def write_csv(path: Path, records: list[dict]):
    """Write one CSV with a header and one row per completed observation."""
    if not records:
        raise ValueError(f"No results to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("Saved:", path)


def plot_lines(path: Path, rows: list[dict], x: str, y: str, group: str,
               title: str, xlabel: str, ylabel: str):
    """Save a group-wise line graph; CSV still works without matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. CSV files were saved; skipping graph.")
        return
    groups = sorted(set(str(row[group]) for row in rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    for value in groups:
        subset = sorted((r for r in rows if str(r[group]) == value),
                        key=lambda r: float(r[x]))
        ax.plot([float(r[x]) for r in subset],
                [float(r[y]) for r in subset], marker="o", label=f"{group}={value}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Saved:", path)


def play_game(config_a: dict, config_b: dict, seat_a: int, chooser,
              game, game_number: int, seed: int):
    """Play one real game with two labeled agents, alternating seats externally.

    chooser(model, player, remaining, config) -> (move, extra_info).
    The statistics for each agent are measured per REAL decision.
    """
    from famnit_gym.envs import mill

    random.seed(seed)
    game.reset()
    labels = {seat_a: "A", 3 - seat_a: "B"}
    configs = {"A": config_a, "B": config_b}
    times = {"A": 0.0, "B": 0.0}
    decisions = {"A": 0, "B": 0}
    max_depth = {"A": 0, "B": 0}
    plies = 0
    winner = "draw"
    while True:
        _, _, terminated, truncated, info = game.last()
        if terminated:
            # Under this project's established terminal convention, info["agent"]
            # identifies the player who lost; the other player is the winner.
            loser = 1 if info["agent"] == "player_1" else 2
            winner = labels[3 - loser]
            break
        if truncated:
            break
        player = 1 if info["agent"] == "player_1" else 2
        label = labels[player]
        model = mill.transition_model(game.unwrapped)
        start = perf_counter()
        move, details = chooser(model, player, remaining_moves(info, player),
                                configs[label])
        times[label] += perf_counter() - start
        decisions[label] += 1
        max_depth[label] = max(max_depth[label], details.get("max_depth", 0))
        game.step(check_move(move, info["legal_moves"]))
        plies += 1
    result = {
        "game": game_number,
        "seat_A": seat_a,
        "winner": winner,
        "plies": plies,
        "A_decisions": decisions["A"],
        "B_decisions": decisions["B"],
        "A_seconds": round(times["A"], 6),
        "B_seconds": round(times["B"], 6),
        "A_max_tree_depth": max_depth["A"],
        "B_max_tree_depth": max_depth["B"],
    }
    return result


def run_pair(config_a: dict, config_b: dict, games_per_seat: int,
             chooser, seed: int, label: str):
    """Alternate player seats and aggregate game outcomes/time into one row."""
    from famnit_gym.envs import mill

    if games_per_seat < 1:
        raise ValueError("games_per_seat must be >= 1")
    game = mill.env(render_mode=None)
    records = []
    try:
        for i in range(2 * games_per_seat):
            # Even index: A plays first. Odd index: B plays first.
            seat_a = 1 if i % 2 == 0 else 2
            result = play_game(config_a, config_b, seat_a, chooser, game,
                               i + 1, seed + i)
            result["matchup"] = label
            records.append(result)
            print(f"{label} | {i+1}/{2*games_per_seat} | winner: {result['winner']}")
    finally:
        game.close()
    n = len(records)
    wins = sum(r["winner"] == "A" for r in records)
    draws = sum(r["winner"] == "draw" for r in records)
    count = sum(r["A_decisions"] for r in records)
    secs = sum(r["A_seconds"] for r in records)
    average_depth = (sum(r["A_max_tree_depth"] for r in records) / n)
    return records, {
        "games": n,
        "wins": wins,
        "losses": n - wins - draws,
        "draws": draws,
        "win_rate_percent": round(100 * wins / n, 2),
        "score_rate_percent": round(100 * (wins + draws * .5) / n, 2),
        "average_ms_per_decision": round(1000 * secs / count, 3) if count else 0,
        "average_max_tree_depth": round(average_depth, 3),
        "decisions": count,
    }
