"""Analyze the MCTS difficulty tournament and create three PNG graphs.


Standalone:
    python analyze_tournament.py

Through run.py (recommended):
    python run.py analyze --input-dir results/difficulties --output-dir results/difficulties
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt

from difficulty_config import LEVEL_ITERATIONS


# The runner sets these variables to the chosen result directories.
# Running the script directly uses the project folder instead.
folder = Path(os.getenv("MILL_INPUT_DIR", str(Path(__file__).resolve().parent)))
output_folder = Path(os.getenv("MILL_OUTPUT_DIR", str(folder)))
output_folder.mkdir(parents=True, exist_ok=True)


# Read the tournament results produced by tournament.py.
with (folder / "tournament_summary.csv").open("r", encoding="utf-8", newline="") as file:
    summary = list(csv.DictReader(file))

with (folder / "tournament_matchups.csv").open("r", encoding="utf-8", newline="") as file:
    matchups = list(csv.DictReader(file))

if not summary or not matchups:
    raise ValueError("Tournament CSV files are empty. Run the difficulties experiment first.")


# Start with all configured difficulties, including Insanity if enabled.
# Also accept old CSVs with a level that is no longer enabled in the config.
games_played = {level: 0 for level in LEVEL_ITERATIONS}

for matchup in matchups:
    level_a = matchup["level_a"]
    level_b = matchup["level_b"]

    number_of_games = (
        int(matchup["level_a_wins"])
        + int(matchup["level_b_wins"])
        + int(matchup["draws"])
    )

    games_played[level_a] = games_played.get(level_a, 0) + number_of_games
    games_played[level_b] = games_played.get(level_b, 0) + number_of_games


# Show difficulties in the order from difficulty_config.py, then any old
# levels that still appear in the CSV data.
summary_by_level = {row["difficulty"]: row for row in summary}
ordered_levels = [level for level in LEVEL_ITERATIONS if level in summary_by_level]
ordered_levels += [level for level in summary_by_level if level not in ordered_levels]


difficulties = []
iterations = []
win_rates = []
thinking_times = []

for difficulty in ordered_levels:
    row = summary_by_level[difficulty]
    wins = int(row["total_wins"])
    total_games = games_played.get(difficulty, 0)
    win_rate = 100 * wins / total_games if total_games else 0

    # Read iterations from the RECORDED result, not today's config value.
    # This avoids incorrectly labeling old experiment results.
    iterations_used = int(row["iterations"])
    average_seconds = float(row["average_thinking_seconds"])

    difficulties.append(difficulty)
    iterations.append(iterations_used)
    win_rates.append(win_rate)
    thinking_times.append(average_seconds * 1000)

    print(
        f"{difficulty}: {iterations_used} iterations, "
        f"{win_rate:.2f}% wins, {average_seconds:.4f} seconds/decision"
    )

    if difficulty in LEVEL_ITERATIONS and iterations_used != LEVEL_ITERATIONS[difficulty]:
        print(
            f"  Note: current config has {LEVEL_ITERATIONS[difficulty]} iterations "
            f"for {difficulty}; this CSV was recorded with {iterations_used}."
        )


# GRAPH 1: Overall win rates
plt.figure(figsize=(8, 5))
plt.bar(difficulties, win_rates)
plt.title("MCTS Tournament Win Rates")
plt.xlabel("Difficulty")
plt.ylabel("Win rate (%)")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig(output_folder / "graph_1_win_rates.png", dpi=150, bbox_inches="tight")
plt.close()


# GRAPH 2: Thinking time vs MCTS iterations
# Sorting by iterations makes the line progress from smaller to larger budgets.
time_points = sorted(zip(iterations, thinking_times))
x_iterations = [point[0] for point in time_points]
y_times = [point[1] for point in time_points]

plt.figure(figsize=(8, 5))
plt.plot(x_iterations, y_times, marker="o")
plt.title("MCTS Iterations vs Thinking Time")
plt.xlabel("MCTS iterations")
plt.ylabel("Average thinking time (ms)")
plt.xticks(sorted(set(x_iterations)))
plt.tight_layout()
plt.savefig(output_folder / "graph_2_thinking_time.png", dpi=150, bbox_inches="tight")
plt.close()


# GRAPH 3: Head-to-head results, automatically supporting additional levels
labels = []
first_wins = []
second_wins = []

for row in matchups:
    labels.append(row["level_a"] + " vs " + row["level_b"])
    first_wins.append(int(row["level_a_wins"]))
    second_wins.append(int(row["level_b_wins"]))

positions = list(range(len(labels)))
left_positions = [x - 0.2 for x in positions]
right_positions = [x + 0.2 for x in positions]

plt.figure(figsize=(max(9, len(labels) * 1.5), 5))
plt.bar(left_positions, first_wins, width=0.4, label="First named AI")
plt.bar(right_positions, second_wins, width=0.4, label="Second named AI")
plt.xticks(positions, labels, rotation=15, ha="right")
plt.title("MCTS Head-to-Head Results")
plt.ylabel("Games won")
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "graph_3_matchups.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nThree graphs saved successfully to:", output_folder)
