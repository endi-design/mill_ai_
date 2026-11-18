
"""Reads the baseline tournament CSV results and draws three graphs.
Run tournament.py first so the input CSV files exist.
"""

import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt


# Find the project folder
folder = Path(os.getenv("MILL_INPUT_DIR", str(Path(__file__).resolve().parent)))
output_folder = Path(os.getenv("MILL_OUTPUT_DIR", str(folder)))
output_folder.mkdir(parents=True, exist_ok=True)


# Read the tournament summary
with open(
    folder / "tournament_summary.csv",
    "r",
    encoding="utf-8"
) as file:
    summary = list(csv.DictReader(file))


# Read the head-to-head results
with open(
    folder / "tournament_matchups.csv",
    "r",
    encoding="utf-8"
) as file:
    matchups = list(csv.DictReader(file))


# Count how many games each difficulty played
games_played = {
    "easy": 0,
    "medium": 0,
    "hard": 0
}

for matchup in matchups:

    level_a = matchup["level_a"]
    level_b = matchup["level_b"]

    number_of_games = (
        int(matchup["level_a_wins"])
        + int(matchup["level_b_wins"])
        + int(matchup["draws"])
    )

    games_played[level_a] += number_of_games
    games_played[level_b] += number_of_games


# Prepare information for graphs
difficulties = []
iterations = []
win_rates = []
thinking_times = []


for row in summary:

    difficulty = row["difficulty"]
    wins = int(row["total_wins"])

    total_games = games_played[difficulty]

    if total_games > 0:
        win_rate = wins / total_games * 100
    else:
        win_rate = 0

    average_seconds = float(
        row["average_thinking_seconds"]
    )

    difficulties.append(difficulty)
    iterations.append(int(row["iterations"]))
    win_rates.append(win_rate)

    # Convert seconds to milliseconds
    thinking_times.append(average_seconds * 1000)

    print(
        difficulty,
        "win rate:", round(win_rate, 2), "%",
        "average time:", round(average_seconds, 4), "s"
    )


# GRAPH 1: Overall win rates
plt.figure(figsize=(8, 5))

plt.bar(difficulties, win_rates)

plt.title("MCTS Tournament Win Rates")
plt.xlabel("Difficulty")
plt.ylabel("Win rate (%)")
plt.ylim(0, 110)

plt.savefig(
    output_folder / "graph_1_win_rates.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# GRAPH 2: Thinking time vs iterations
plt.figure(figsize=(8, 5))

plt.plot(iterations, thinking_times, marker="o")

plt.title("MCTS Iterations vs Thinking Time")
plt.xlabel("MCTS iterations")
plt.ylabel("Average thinking time (ms)")
plt.xticks(iterations)

plt.savefig(
    output_folder / "graph_2_thinking_time.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# GRAPH 3: Head-to-head results
labels = []
first_wins = []
second_wins = []

for row in matchups:

    labels.append(
        row["level_a"] + " vs " + row["level_b"]
    )

    first_wins.append(int(row["level_a_wins"]))
    second_wins.append(int(row["level_b_wins"]))


positions = list(range(len(labels)))

left_positions = [x - 0.2 for x in positions]
right_positions = [x + 0.2 for x in positions]

plt.figure(figsize=(9, 5))

plt.bar(
    left_positions,
    first_wins,
    width=0.4,
    label="First named AI"
)

plt.bar(
    right_positions,
    second_wins,
    width=0.4,
    label="Second named AI"
)

plt.xticks(positions, labels)
plt.title("MCTS Head-to-Head Results")
plt.ylabel("Games won")
plt.legend()

plt.savefig(
    output_folder / "graph_3_matchups.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print()
print("Three graphs saved successfully!")
