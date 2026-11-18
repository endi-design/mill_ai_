"""Command-line control panel for the FAMNIT Mill MCTS seminar project.

Use `python run.py --help` to see available actions. This file reuses
expansion_practice.py, the supplied experiment scripts, and human_vs_mcts.py.
"""

import argparse
import csv
import os
from pathlib import Path
import random
import runpy
import sys
from time import perf_counter

from difficulty_config import LEVEL_ITERATIONS, DEFAULT_C

PROJECT_DIR = Path(__file__).resolve().parent
LEVEL_NAMES = ("random", "mcts", *LEVEL_ITERATIONS)
STUDY_SCRIPTS = {
    "difficulties": "tournament.py",
    "random-moves": "random_moves_experiment.py",
    "time-budget": "time_budget_experiment.py",
    "c-constant": "c_constant_experiment.py",
    "uct-mcts": "c_constant_experiment.py",  # A readable alias.
}


def remaining_moves_from_info(info, player):
    """Estimate remaining individual turns from FAMNIT's 100-round counter."""
    round_number = max(1, min(100, int(info["move"])))
    return max(0, 2 * (101 - round_number) - (1 if player == 2 else 0))


def verify_move(move, legal_moves):
    """Convert move numbers to ordinary integers and reject illegal actions.

    FAMNIT silently substitutes random legal moves for illegal actions,
    so checking legality is essential for honest experiments.
    """
    selected = [int(x) for x in move]
    legal = [[int(x) for x in action] for action in legal_moves]
    if selected not in legal:
        raise ValueError(f"Illegal AI move {selected}; legal moves: {legal}")
    return selected


def move_for_agent(agent, model, player, remaining, *, iterations, C):
    """Choose a move with Random or the user's existing MCTS implementation."""
    legal = model.legal_moves(player)
    if not legal:
        raise ValueError("No legal actions in a nonterminal position")
    if agent == "random":
        return random.choice(legal), 0
    from expansion_practice import choose_mcts_move
    count = LEVEL_ITERATIONS.get(agent, iterations)
    selected = choose_mcts_move(model, player, count, C, remaining)
    if selected is None:
        raise RuntimeError("MCTS returned no move")
    return selected, 0 if len(legal) == 1 else count


def play_games(args):
    """Run Random/MCTS matchups, optionally swapping starting seats each game.

    `--games-per-seat N` alternates seats for 2*N games. Without it,
    `--games N` keeps the requested Player 1 / Player 2 agents fixed.
    """
    from famnit_gym.envs import mill

    random.seed(args.seed)
    total_games = args.games_per_seat * 2 if args.games_per_seat else args.games
    if total_games < 1:
        raise ValueError("At least one game is required")
    stats = {"p1": {"wins": 0, "moves": 0, "seconds": 0.0},
             "p2": {"wins": 0, "moves": 0, "seconds": 0.0}}
    rows = []
    draws = 0
    game = mill.env(render_mode=None)
    try:
        for i in range(total_games):
            game.reset()
            swapped = bool(args.games_per_seat and i % 2)
            seats = {1: "p2" if swapped else "p1", 2: "p1" if swapped else "p2"}
            winner = "draw"
            plies = 0
            while True:
                _, _, terminated, truncated, info = game.last()
                if terminated:
                    # The current agent is the loser (as documented by FAMNIT).
                    losing_seat = 1 if info["agent"] == "player_1" else 2
                    winner = seats[3 - losing_seat]
                    stats[winner]["wins"] += 1
                    break
                if truncated:
                    draws += 1
                    break
                player = 1 if info["agent"] == "player_1" else 2
                label = seats[player]
                agent = args.p1 if label == "p1" else args.p2
                iterations = args.iterations1 if label == "p1" else args.iterations2
                C = args.c1 if label == "p1" else args.c2
                model = mill.transition_model(game.unwrapped)
                remaining = remaining_moves_from_info(info, player)
                start = perf_counter()
                chosen, _ = move_for_agent(agent, model, player, remaining,
                                           iterations=iterations, C=C)
                elapsed = perf_counter() - start
                stats[label]["moves"] += 1
                stats[label]["seconds"] += elapsed
                game.step(verify_move(chosen, info["legal_moves"]))
                plies += 1
            rows.append({"game": i + 1, "player1_agent": args.p2 if swapped else args.p1,
                         "player2_agent": args.p1 if swapped else args.p2,
                         "winner_label": winner, "plies": plies})
            print(f"Game {i + 1}/{total_games}: {winner}")
    finally:
        game.close()

    print("\nMatch summary:")
    for label, agent in (("p1", args.p1), ("p2", args.p2)):
        s = stats[label]
        average = s["seconds"] / s["moves"] if s["moves"] else 0.0
        print(f"  {label} ({agent}): {s['wins']} wins, {s['moves']} decisions, "
              f"{average:.4f} seconds/decision")
    print(f"  Draws: {draws}")
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "match_games.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", output / "match_games.csv")


def run_legacy(script, *, games_per_matchup=None, seed=None, output_dir=None,
               input_dir=None):
    """Invoke an existing experiment script with user-selected parameters.

    Configuration travels via environment variables so the standalone scripts
    keep working normally while the command-line interface can override them.
    """
    settings = {"MILL_GAMES_PER_MATCHUP": games_per_matchup,
                "MILL_SEED": seed,
                "MILL_OUTPUT_DIR": output_dir,
                "MILL_INPUT_DIR": input_dir}
    before = {key: os.environ.get(key) for key in settings}
    try:
        for key, value in settings.items():
            if value is not None:
                os.environ[key] = str(value)
        runpy.run_path(str(PROJECT_DIR / script), run_name="__main__")
    finally:
        for key, previous in before.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def run_study(args):
    """Run the selected study with swapped starting seats."""
    if args.study == "uct-random":
        # Unlike the direct C-vs-C test, compare each C configuration to Random.
        from experiment_runner import run_uct_vs_random
        run_uct_vs_random(args.games_per_seat, args.iterations,
                          args.seed, Path(args.output_dir))
        return
    script = STUDY_SCRIPTS[args.study]
    run_legacy(script, games_per_matchup=2 * args.games_per_seat,
               seed=args.seed, output_dir=args.output_dir)
    print("Finished:", args.study, "— output folder:", args.output_dir)


def run_human(args):
    """Launch the graphical wrapper; game controls are handled in that module."""
    from human_vs_mcts import main
    main(difficulty=args.difficulty, human_seat=args.human_seat)


def run_smoke():
    """Test that FAMNIT is installed and MCTS can choose a legal opening move."""
    from famnit_gym.envs import mill
    from expansion_practice import choose_mcts_move
    game = mill.env(render_mode=None)
    try:
        game.reset()
        _, _, terminated, truncated, info = game.last()
        if terminated or truncated:
            raise RuntimeError("A fresh board unexpectedly ended")
        player = 1 if info["agent"] == "player_1" else 2
        model = mill.transition_model(game.unwrapped)
        remaining = remaining_moves_from_info(info, player)
        action = choose_mcts_move(model, player, 5, DEFAULT_C, remaining)
        print("MCTS opening move:", verify_move(action, info["legal_moves"]))
        game.step(verify_move(action, info["legal_moves"]))
        print("Smoke test passed: MCTS selected and played a legal move.")
    finally:
        game.close()


def print_configuration():
    """Show the profiles that human play, matches and tournament really use."""
    print(f"Normal UCT exploration constant C: {DEFAULT_C}")
    for level, iterations in LEVEL_ITERATIONS.items():
        print(f"  {level}: {iterations} iterations")


def build_parser():
    """Describe the available terminal commands and their arguments."""
    p = argparse.ArgumentParser(description="FAMNIT Mill MCTS seminar experiments")
    sub = p.add_subparsers(dest="command", required=True)

    match = sub.add_parser("match", help="Play MCTS vs Random or MCTS vs MCTS")
    match.add_argument("--p1", choices=LEVEL_NAMES, default="mcts")
    match.add_argument("--p2", choices=LEVEL_NAMES, default="random")
    match.add_argument("--iterations1", type=int, default=100)
    match.add_argument("--iterations2", type=int, default=100)
    match.add_argument("--c1", type=float, default=DEFAULT_C)
    match.add_argument("--c2", type=float, default=DEFAULT_C)
    match.add_argument("--games", type=int, default=1)
    match.add_argument("--games-per-seat", type=int, default=None,
                       help="N games per starting seat (total 2*N), overrides --games")
    match.add_argument("--seed", type=int, default=42)
    match.add_argument("--output-dir", default=None)
    match.set_defaults(action=play_games)

    exp = sub.add_parser("experiment", help="Run difficulty experiments (all pairs for configured levels)")
    exp.add_argument("--study", choices=sorted([*STUDY_SCRIPTS, "uct-random"]),
                     required=True)
    exp.add_argument("--games-per-seat", type=int, default=5,
                     help="5 means 10 games per matchup and 30 games total")
    exp.add_argument("--iterations", type=int, default=100,
                     help="MCTS iterations for uct-random experiment")
    exp.add_argument("--seed", type=int, default=20261117)
    exp.add_argument("--output-dir", default="results")
    exp.set_defaults(action=run_study)

    config_cmd = sub.add_parser("config", help="Show the effective shared difficulty settings")
    config_cmd.set_defaults(action=lambda _: print_configuration())

    human = sub.add_parser("human", help="Play through the FAMNIT graphical interface")
    human.add_argument("--difficulty", choices=LEVEL_ITERATIONS, default="medium")
    human.add_argument("--human-seat", type=int, choices=[1, 2], default=1)
    human.set_defaults(action=run_human)

    analyze = sub.add_parser("analyze", help="Regenerate PNG graphs from baseline CSV")
    analyze.add_argument("--input-dir", default=".")
    analyze.add_argument("--output-dir", default=None)
    analyze.set_defaults(action=lambda a: run_legacy(
        "analyze_tournament.py", input_dir=a.input_dir,
        output_dir=a.output_dir or a.input_dir))

    smoke = sub.add_parser("smoke", help="Verify FAMNIT and MCTS on one opening move")
    smoke.set_defaults(action=lambda _: run_smoke())
    return p


def main(argv=None):
    """Parse arguments, run the requested action, and show useful errors."""
    args = build_parser().parse_args(argv)
    if hasattr(args, "games_per_seat") and args.games_per_seat is not None:
        if args.games_per_seat < 1:
            raise SystemExit("--games-per-seat must be at least 1")
    if hasattr(args, "iterations") and args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if hasattr(args, "iterations1") and (args.iterations1 < 1 or args.iterations2 < 1):
        raise SystemExit("--iterations1 and --iterations2 must be at least 1")
    args.action(args)


if __name__ == "__main__":
    main()
