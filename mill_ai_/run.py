"""Simple command-line entry point for the entire seminar project.

Usage examples:
  python run.py match --p1 mcts --p2 random --iterations1 100
  python run.py match --p1 mcts --p2 mcts --c1 0.5 --c2 2
  python run.py match --p1 easy --p2 hard
  python run.py experiment --study uct-random --games-per-seat 2
  python run.py experiment --study difficulties --games-per-seat 2
  python run.py human --difficulty easy --human-seat 1

Everything except FAMNIT Gym itself uses Python's standard library.
"""

from __future__ import annotations # It allows Python to store type annotations without immediately evaluating them

import argparse # built-in Python library that allows you to pass arguments to your program through the terminal
import math
from agents import DIFFICULTY_PRESETS
from experiments import DEFAULT_C_VALUES, run_experiments
from game import play_match


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nine Men's Morris — MCTS Intelligent Systems seminar"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ----- ONE MATCH -----
    match = sub.add_parser("match", help="Play one AI vs random or AI vs AI match")
    names = ["random", "mcts", *DIFFICULTY_PRESETS]
    match.add_argument("--p1", choices=names, default="mcts",
                       help="Algorithm for player 1")
    match.add_argument("--p2", choices=names, default="random",
                       help="Algorithm for player 2")
    match.add_argument("--iterations1", type=int, default=150,
                       help="MCTS iterations for p1 when p1=mcts")
    match.add_argument("--iterations2", type=int, default=150,
                       help="MCTS iterations for p2 when p2=mcts")
    match.add_argument("--c1", type=float, default=math.sqrt(2),
                       help="UCT exploration constant for p1")
    match.add_argument("--c2", type=float, default=math.sqrt(2),
                       help="UCT exploration constant for p2")
    match.add_argument("--seed", type=int, default=42)
    match.add_argument("--show-board", action="store_true",
                       help="Print the ASCII board between moves")
    match.add_argument("--quiet", action="store_true",
                       help="Print only final result")

    # ----- EXPERIMENTS -----
    experiment = sub.add_parser("experiment", help="Run UCT or difficulty studies")
    experiment.add_argument("--study", required=True,
                            choices=["uct-random", "uct-mcts", "difficulties"])
    experiment.add_argument("--games-per-seat", type=int, default=1,
                            help="Number of matches for each side of a comparison")
    experiment.add_argument("--iterations", type=int, default=60,
                            help="Search simulations for tested MCTS")
    experiment.add_argument("--baseline-iterations", type=int, default=60,
                            help="Search simulations for reference MCTS")
    experiment.add_argument("--c-values", type=float, nargs="+",
                            default=list(DEFAULT_C_VALUES),
                            help="UCT coefficients to compare")
    experiment.add_argument("--seed", type=int, default=2026)
    experiment.add_argument("--output-dir", default="results")

    # ----- HUMAN vs AI -----
    human = sub.add_parser("human", help="Play against MCTS with the GUI wrapper")
    human.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS) + ["mcts"],
                       default="easy", help="AI opponent difficulty")
    human.add_argument("--human-seat", type=int, choices=[1, 2], default=1)
    human.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "match":
        play_match(
            player1=args.p1, player2=args.p2,
            iterations1=args.iterations1, iterations2=args.iterations2,
            exploration1=args.c1, exploration2=args.c2,
            seed=args.seed, show_board=args.show_board,
            verbose=not args.quiet,
        )
    elif args.command == "experiment":
        run_experiments(
            study=args.study, games_per_seat=args.games_per_seat,
            iterations=args.iterations,
            baseline_iterations=args.baseline_iterations,
            seed=args.seed, c_values=tuple(args.c_values),
            output_dir=args.output_dir,
        )
    elif args.command == "human":
        from human_play import play_human
        play_human(difficulty=args.difficulty, human_seat=args.human_seat,
                   seed=args.seed)


if __name__ == "__main__":
    main()
