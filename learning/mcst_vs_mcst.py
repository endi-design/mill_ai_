"""Legacy comparison: Medium MCTS plays Hard MCTS, alternating seats.
For all experiments with terminal arguments, prefer run.py.

Notes: This file is part of the FAMNIT Mill seminar project.
"""


from famnit_gym.envs import mill
from expansion_practice import choose_mcts_move
from time import perf_counter
import os


# AI difficulty settings
DIFFICULTIES = {
    "easy": 10,
    "medium": 50,
    "hard": 100
}

number_of_games = int(os.getenv("MILL_GAMES", "10"))

# Experiment results
medium_wins = 0
hard_wins = 0
draws = 0

# Timing information
total_time = {
    "medium": 0,
    "hard": 0
}

moves_count = {
    "medium": 0,
    "hard": 0
}


def get_remaining_moves(info, player):
    """Convert the FAMNIT round counter into the number of individual moves
    left until the 100-round game limit (Player 2 has one fewer)."""

    round_number = int(info["move"])

    remaining = (101 - round_number) * 2

    if player == 2:
        remaining = remaining - 1

    return remaining


game = mill.env(render_mode=None)

try:

    for game_number in range(number_of_games):

        game.reset()

        # STEP 52: Alternate which AI plays first
        if game_number % 2 == 0:
            player1_difficulty = "medium"
            player2_difficulty = "hard"
        else:
            player1_difficulty = "hard"
            player2_difficulty = "medium"

        print()
        print("Starting game", game_number + 1)
        print("Player 1:", player1_difficulty)
        print("Player 2:", player2_difficulty)

        while True:

            observation, reward, terminated, truncated, info = game.last()

            # Check whether someone has lost
            if terminated:

                # Same terminal convention as your previous code:
                # info["agent"] identifies the losing player.
                if info["agent"] == "player_1":
                    winner = 2
                    winning_difficulty = player2_difficulty
                else:
                    winner = 1
                    winning_difficulty = player1_difficulty

                if winning_difficulty == "medium":
                    medium_wins += 1
                else:
                    hard_wins += 1

                print(
                    "Game", game_number + 1,
                    ":", winning_difficulty,
                    "won as Player", winner
                )

                break

            # Game reached its limit
            if truncated:

                draws += 1

                print("Game", game_number + 1, ": Draw")

                break

            # Determine whose turn it is
            if info["agent"] == "player_1":
                player = 1
                difficulty = player1_difficulty
            else:
                player = 2
                difficulty = player2_difficulty

            # Get the iteration count
            iterations = DIFFICULTIES[difficulty]

            # Get the current board
            model = mill.transition_model(game.unwrapped)

            # Calculate the number of remaining moves
            remaining_moves = get_remaining_moves(info, player)

            # Start measuring thinking time
            start_time = perf_counter()

            # Both players use MCTS
            chosen_move = choose_mcts_move(
                model,
                player,
                iterations,
                1.4,
                remaining_moves
            )

            # Stop measuring thinking time
            thinking_time = perf_counter() - start_time

            # Save timing results by difficulty
            total_time[difficulty] += thinking_time
            moves_count[difficulty] += 1

            # Convert the chosen move to Python integers
            move_for_game = []

            for number in chosen_move:
                move_for_game.append(int(number))

            # Convert all legal moves to Python integers
            legal_moves = []

            for move in info["legal_moves"]:
                legal_moves.append([int(x) for x in move])

            # Check that MCTS selected a valid move
            if move_for_game not in legal_moves:
                raise ValueError("MCTS selected an illegal move")

            # Play the move
            game.step(move_for_game)

finally:
    game.close()


# Final experiment results
print()
print("----------------------------")
print("MCTS VS MCTS RESULTS")
print("----------------------------")

print("Total games:", number_of_games)
print("Medium wins:", medium_wins)
print("Hard wins:", hard_wins)
print("Draws:", draws)

print()

for difficulty in ["medium", "hard"]:

    if moves_count[difficulty] > 0:

        average_time = (
            total_time[difficulty] / moves_count[difficulty]
        )

        print(
            difficulty,
            "average thinking time:",
            round(average_time, 4),
            "seconds"
        )

print("----------------------------")

