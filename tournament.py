"""Baseline MCTS difficulty tournament: Easy, Medium and Hard differ
only in search iteration counts. Exports CSV files for graphs and analysis.

Notes: This file is part of the FAMNIT Mill seminar project.
"""


import csv
import os
import random
from pathlib import Path
from time import perf_counter

from famnit_gym.envs import mill
from expansion_practice import choose_mcts_move
from difficulty_config import LEVEL_ITERATIONS as DIFFICULTIES, DEFAULT_C
from itertools import combinations


# All iteration difficulty settings are imported from difficulty_config.py.
# plays 50 mill games
GAMES_PER_MATCHUP = int(os.getenv("MILL_GAMES_PER_MATCHUP", "25"))

MATCHUPS = list(combinations(DIFFICULTIES, 2))


# Overall tournament statistics
total_wins = {level: 0 for level in DIFFICULTIES}

total_draws = {level: 0 for level in DIFFICULTIES}

total_time = {level: 0.0 for level in DIFFICULTIES}

moves_count = {level: 0 for level in DIFFICULTIES}


# Lists for CSV export
game_results = []
matchup_results = []


def get_remaining_moves(info, player):
    """Convert the FAMNIT round counter into the number of individual moves
    left until the 100-round game limit (Player 2 has one fewer)."""

    round_number = int(info["move"])

    remaining = (101 - round_number) * 2

    if player == 2:
        remaining -= 1

    return max(0, remaining)


# Seed Python random decisions so the experiment can be repeated.
# The game environment may have separate random generators.
random.seed(int(os.getenv("MILL_SEED", "20261117")))

# Create the environment
game = mill.env(render_mode=None)

try:

    for matchup in MATCHUPS:

        level_a = matchup[0]
        level_b = matchup[1]

        wins = {
            level_a: 0,
            level_b: 0
        }

        draws = 0

        print()
        print("=========================")
        print(level_a, "vs", level_b)
        print("=========================")

        for game_number in range(1, GAMES_PER_MATCHUP + 1):

            game.reset()

            # Alternate who plays first
            if game_number % 2 == 1:
                player1_difficulty = level_a
                player2_difficulty = level_b
            else:
                player1_difficulty = level_b
                player2_difficulty = level_a

            turns_played = 0

            game_times = {
                level_a: 0.0,
                level_b: 0.0
            }

            winner_difficulty = "draw"

            while True:

                observation, reward, terminated, truncated, info = game.last()

                # Check for a winner
                if terminated:

                    # The current setup reports the losing
                    # player through info["agent"]
                    if info["agent"] == "player_1":
                        winner_difficulty = player2_difficulty
                    else:
                        winner_difficulty = player1_difficulty

                    wins[winner_difficulty] += 1
                    total_wins[winner_difficulty] += 1

                    break

                # Check for a draw
                if truncated:

                    draws += 1
                    total_draws[level_a] += 1
                    total_draws[level_b] += 1

                    break

                # Determine which AI is playing
                if info["agent"] == "player_1":
                    player = 1
                    difficulty = player1_difficulty
                else:
                    player = 2
                    difficulty = player2_difficulty

                # MCTS settings
                iterations = DIFFICULTIES[difficulty]

                model = mill.transition_model(game.unwrapped)

                remaining_moves = get_remaining_moves(
                    info, player
                )

                # Start timing
                start_time = perf_counter()

                # Choose an MCTS move
                chosen_move = choose_mcts_move(
                    model,
                    player,
                    iterations,
                    DEFAULT_C,
                    remaining_moves
                )

                # Stop timing
                thinking_time = perf_counter() - start_time

                # Record thinking time
                total_time[difficulty] += thinking_time
                moves_count[difficulty] += 1
                game_times[difficulty] += thinking_time

                # Convert MCTS move to integers
                move_for_game = [
                    int(number)
                    for number in chosen_move
                ]

                # Convert legal moves to integers
                legal_moves = [
                    [int(number) for number in move]
                    for move in info["legal_moves"]
                ]

                # Verify the move is legal
                if move_for_game not in legal_moves:
                    raise ValueError("MCTS selected an illegal move")

                # Play the move
                game.step(move_for_game)

                turns_played += 1

            # Save one row after each completed game
            game_results.append({
                "matchup": level_a + "_vs_" + level_b,
                "game_number": game_number,
                "player1": player1_difficulty,
                "player2": player2_difficulty,
                "winner": winner_difficulty,
                "moves_played": turns_played,
                "player1_thinking_seconds": round(
                    game_times[player1_difficulty], 6
                ),
                "player2_thinking_seconds": round(
                    game_times[player2_difficulty], 6
                )
            })

            print(
                "Game", game_number,
                ":", winner_difficulty
            )

        # Save a summary for this matchup
        matchup_results.append({
            "matchup": level_a + "_vs_" + level_b,
            "level_a": level_a,
            "level_b": level_b,
            "level_a_wins": wins[level_a],
            "level_b_wins": wins[level_b],
            "draws": draws
        })

        print()
        print("RESULTS:")
        print(level_a, "wins:", wins[level_a])
        print(level_b, "wins:", wins[level_b])
        print("Draws:", draws)

finally:
    game.close()


# FINAL TOURNAMENT RESULTS
print()
print("=========================")
print("FINAL TOURNAMENT RESULTS")
print("Shared UCT constant C:", DEFAULT_C)
print("=========================")

for difficulty in DIFFICULTIES:

    if moves_count[difficulty] > 0:
        average_time = (
            total_time[difficulty] / moves_count[difficulty]
        )
    else:
        average_time = 0.0

    print()
    print("Difficulty:", difficulty)
    print("Iterations:", DIFFICULTIES[difficulty])
    print("Total wins:", total_wins[difficulty])
    print("Total moves:", moves_count[difficulty])
    print(
        "Average thinking time:",
        round(average_time, 4),
        "seconds"
    )


# CSV EXPORT
# Save files in the same folder as tournament.py
output_folder = Path(os.getenv("MILL_OUTPUT_DIR", str(Path(__file__).resolve().parent)))
output_folder.mkdir(parents=True, exist_ok=True)


# CSV 1: Every individual game
with open(
    output_folder / "tournament_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    columns = [
        "matchup",
        "game_number",
        "player1",
        "player2",
        "winner",
        "moves_played",
        "player1_thinking_seconds",
        "player2_thinking_seconds"
    ]

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(game_results)


# CSV 2: Results for each matchup
with open(
    output_folder / "tournament_matchups.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    columns = [
        "matchup",
        "level_a",
        "level_b",
        "level_a_wins",
        "level_b_wins",
        "draws"
    ]

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(matchup_results)


# CSV 3: Overall statistics by difficulty
with open(
    output_folder / "tournament_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    columns = [
        "difficulty",
        "iterations",
        "total_wins",
        "total_draws",
        "total_moves",
        "average_thinking_seconds"
    ]

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for difficulty in DIFFICULTIES:

        if moves_count[difficulty] > 0:
            average_time = (
                total_time[difficulty] / moves_count[difficulty]
            )
        else:
            average_time = 0.0

        writer.writerow({
            "difficulty": difficulty,
            "iterations": DIFFICULTIES[difficulty],
            "total_wins": total_wins[difficulty],
            "total_draws": total_draws[difficulty],
            "total_moves": moves_count[difficulty],
            "average_thinking_seconds": round(
                average_time, 6
            )
        })


print()
print("CSV files saved:")
print("- tournament_results.csv")
print("- tournament_matchups.csv")
print("- tournament_summary.csv")
