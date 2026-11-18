"""Study the UCT exploration constant C using actual FAMNIT Mill games.

Two studies are supported:
  tournament: all configured difficulty agents use the SAME C in each tournament,
              but have different MCTS iteration budgets.
  fixed:      vary C of one MCTS agent against an unchanged MCTS opponent.

This distinguishes C parameter tuning from iteration-based difficulty.
Run `python mcts_parameter_experiments/c_study.py --help` for options.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations
from common import run_pair, write_csv, plot_lines
from difficulty_config import (
    LEVEL_ITERATIONS,
    C_STUDY_VALUES,
    C_STUDY_ITERATIONS,
    DEFAULT_C,
)

# Named levels, including Insanity if enabled, come from difficulty_config.py.
LEVELS = dict(LEVEL_ITERATIONS)
PAIRS = list(combinations(LEVELS, 2))


def choose_move(model, player, remaining, setting):
    """Use the SAME existing MCTS code, changing only settings in config."""
    from expansion_practice import choose_mcts_move
    legal = model.legal_moves(player)
    if not legal:
        raise ValueError("No legal move in nonterminal position")
    if len(legal) == 1:
        return legal[0], {"max_depth": 0}
    move = choose_mcts_move(model, player, setting["iterations"],
                            setting["C"], remaining)
    return move, {"max_depth": 0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["tournament", "fixed"],
                        default="tournament")
    parser.add_argument("--c-values", type=float, nargs="+",
                        default=sorted({0.0, *C_STUDY_VALUES.values()}),
                        help="C values to test; defaults come from difficulty_config.py plus 0.0")
    parser.add_argument("--games-per-seat", type=int, default=3,
                        help="Each matchup has 2 x games-per-seat games")
    # Overrides are optional. Without them, difficulty_config.py is authoritative.
    # This automatically adds --insanity when that difficulty is enabled.
    for level, iterations in LEVELS.items():
        parser.add_argument(
            f"--{level}", type=int, default=None,
            help=f"Override {level} iterations (config default: {iterations})",
        )
    parser.add_argument("--iterations", type=int, default=C_STUDY_ITERATIONS,
                        help="Candidate iterations for --study fixed")
    parser.add_argument("--opponent-iterations", type=int,
                        default=C_STUDY_ITERATIONS)
    parser.add_argument("--opponent-c", type=float, default=DEFAULT_C)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "c_study")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(c < 0 for c in args.c_values) or args.opponent_c < 0:
        raise ValueError("C must be nonnegative")
    budgets = dict(LEVELS)
    for level in LEVELS:
        override = getattr(args, level)
        if override is not None:
            budgets[level] = override
    if min(*budgets.values(), args.iterations, args.opponent_iterations) < 1:
        raise ValueError("All iteration budgets must be positive")
    if args.games_per_seat < 1:
        raise ValueError("--games-per-seat must be at least 1")
    if args.study == "tournament":
        print("Iteration budgets from difficulty_config.py / CLI overrides:", budgets)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, game_rows = [], []
    for c_index, C in enumerate(args.c_values):
        if args.study == "tournament":
            for pair_index, (level_a, level_b) in enumerate(PAIRS):
                label = f"C={C} {level_a} vs {level_b}"
                games, stats_a = run_pair(
                    {"iterations": budgets[level_a], "C": C},
                    {"iterations": budgets[level_b], "C": C},
                    args.games_per_seat, choose_move,
                    args.seed + c_index * 100000 + pair_index * 1000, label)
                for r in games:
                    r.update({"C": C, "level_A": level_a, "level_B": level_b,
                              "iterations_A": budgets[level_a],
                              "iterations_B": budgets[level_b]})
                game_rows.extend(games)
                # Aggregate both sides of each pair, with their timing.
                stats_b = {
                    "games": stats_a["games"],
                    "wins": stats_a["losses"],
                    "losses": stats_a["wins"],
                    "draws": stats_a["draws"],
                    "win_rate_percent": round(100*stats_a["losses"]/stats_a["games"],2),
                    "score_rate_percent": round(100*(stats_a["losses"]+stats_a["draws"]*.5)/stats_a["games"],2),
                    "average_ms_per_decision": round(1000*sum(r["B_seconds"] for r in games) / sum(r["B_decisions"] for r in games),3) if sum(r["B_decisions"] for r in games) else 0,
                    "average_max_tree_depth": round(sum(r["B_max_tree_depth"] for r in games)/len(games),3),
                    "decisions": sum(r["B_decisions"] for r in games),
                }
                summary_rows.append({"study": args.study, "C": C,
                                     "difficulty": level_a,
                                     "iterations": budgets[level_a],
                                     "opponent": level_b, **stats_a})
                summary_rows.append({"study": args.study, "C": C,
                                     "difficulty": level_b,
                                     "iterations": budgets[level_b],
                                     "opponent": level_a, **stats_b})
        else:
            label = f"candidate C={C} vs fixed C={args.opponent_c}"
            games, stats = run_pair(
                {"iterations": args.iterations, "C": C},
                {"iterations": args.opponent_iterations, "C": args.opponent_c},
                args.games_per_seat, choose_move,
                args.seed + c_index * 100000, label)
            for r in games:
                r.update({"C": C, "level_A": "candidate", "level_B": "fixed",
                          "iterations_A": args.iterations,
                          "iterations_B": args.opponent_iterations})
            game_rows.extend(games)
            summary_rows.append({"study": args.study, "C": C,
                                 "difficulty": "candidate",
                                 "iterations": args.iterations,
                                 "opponent": f"C={args.opponent_c}", **stats})

    write_csv(out / "c_games.csv", game_rows)
    write_csv(out / "c_summary.csv", summary_rows)
    if args.study == "tournament":
        # Add a convenience table with each difficulty's TOTAL across two opponents.
        totals = []
        for C in args.c_values:
            for difficulty in LEVELS:
                matches = [r for r in summary_rows if r["C"] == C and r["difficulty"] == difficulty]
                games = sum(r["games"] for r in matches)
                wins = sum(r["wins"] for r in matches)
                draws = sum(r["draws"] for r in matches)
                weighted_ms = sum(r["average_ms_per_decision"] * r["decisions"] for r in matches)
                decisions = sum(r["decisions"] for r in matches)
                totals.append({"C": C, "difficulty": difficulty,
                               "iterations": matches[0]["iterations"],
                               "games": games, "wins": wins, "draws": draws,
                               "win_rate_percent": round(100*wins/games, 2),
                               "score_rate_percent": round(100*(wins+0.5*draws)/games, 2),
                               "average_ms_per_decision": round(weighted_ms/decisions,3) if decisions else 0})
        write_csv(out / "c_tournament_totals.csv", totals)
        plot_lines(out / "c_win_rate.png", totals, "C", "win_rate_percent",
                   "difficulty", "Which difficulty wins as C changes?", "C", "Win rate (%)")
        plot_lines(out / "c_thinking_time.png", totals, "C", "average_ms_per_decision",
                   "difficulty", "Thinking time vs C", "C", "Milliseconds / decision")
        for r in totals:
            print(f"C={r['C']:>3} {r['difficulty']:<6}: {r['wins']}/{r['games']} wins, {r['win_rate_percent']}%, {r['average_ms_per_decision']} ms")
    else:
        plot_lines(out / "c_win_rate.png", summary_rows, "C", "score_rate_percent",
                   "difficulty", "C vs the same opponent", "C", "Score rate (win + 1/2 draw) %")
        plot_lines(out / "c_thinking_time.png", summary_rows, "C", "average_ms_per_decision",
                   "difficulty", "Thinking time vs C", "C", "Milliseconds / decision")
        for r in summary_rows:
            print(f"C={r['C']:>3}: {r['wins']}/{r['games']} wins, {r['draws']} draws, {r['average_ms_per_decision']} ms")


if __name__ == "__main__":
    main()
