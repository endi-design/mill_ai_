"""Runs a fixed-difficulty MCTS agent against a random opponent.
Counts wins and measures the AI's average decision time.

Notes: This file is part of the FAMNIT Mill seminar project.
"""

from famnit_gym.envs import mill
from expansion_practice import choose_mcts_move
import random
import os
from time import perf_counter
DIFFICULTIES = {
    "easy": 10,
    "medium": 50,
    "hard": 100
}

difficulty = os.getenv("MILL_DIFFICULTY", "hard")

iterations = DIFFICULTIES[difficulty]
game = mill.env(render_mode=None)

mcts_wins = 0
random_wins = 0
draws = 0

total_thinking_time = 0
mcts_moves = 0
def get_remaining_moves(info, player):
    """Convert the FAMNIT round counter into the number of individual moves
    left until the 100-round game limit (Player 2 has one fewer)."""

    round_number = int(info["move"])

    remaining = (101 - round_number) * 2

    if player == 2:
        remaining = remaining - 1

    return remaining
print(get_remaining_moves({"move": 1}, 1))
print(get_remaining_moves({"move": 1}, 2))

print(get_remaining_moves({"move": 100}, 1))
print(get_remaining_moves({"move": 100}, 2))
for game_number in range(int(os.getenv("MILL_GAMES", "10"))):

    game.reset()
    turns_played = 0

    while True:

        observation, reward, terminated, truncated, info = game.last()

        if terminated:
            if info["agent"] == "player_1":
                random_wins += 1
                print("Game", game_number + 1, ": Random won")
            else:
                mcts_wins += 1
                print("Game", game_number + 1, ": MCTS won")
            break

        if truncated:
            draws += 1
            print("Game", game_number + 1, ": Draw")
            break

        if info["agent"] == "player_1":
            player = 1
        else:
            player = 2

        if player == 1:
            model = mill.transition_model(game.unwrapped)

            remaining_moves = get_remaining_moves(info, player)

            start_time = perf_counter()

            chosen_move = choose_mcts_move(
                model, player, iterations, 1.4, remaining_moves
            )

            end_time = perf_counter()

            thinking_time = end_time - start_time
            total_thinking_time = total_thinking_time + thinking_time

            mcts_moves = mcts_moves + 1
           # print("MCTS thinking time:", thinking_time, "seconds")
        else:
            chosen_move = random.choice(info["legal_moves"])

        move_for_game = []

        for number in chosen_move:
            move_for_game.append(int(number))

        legal_moves = []
        for move in info["legal_moves"]:
            legal_moves.append([int(x) for x in move])

        if move_for_game not in legal_moves:
            raise ValueError("Illegal move")

        game.step(move_for_game)

        turns_played += 1

if mcts_moves > 0:

    average_time = total_thinking_time / mcts_moves

    print("----------------------")
    print("Difficulty:", difficulty)
    print("Iterations:", iterations)
    print("MCTS wins:", mcts_wins)
    print("Random wins:", random_wins)
    print("Draws:", draws)
    print("Average seconds per move:", average_time)
    print("----------------------")
game.close()
