"""Learning exercise: tests a basic Monte Carlo rollout agent against Random.
Despite the historical filename, Player 1 uses a simple Monte Carlo agent, not Random.

Notes: This file is part of the FAMNIT Mill seminar project.
"""

from famnit_gym.envs import mill
import random



def choose_random_move(legal_moves): # from all the legal moves chooses the random move
    """Return one uniformly random move from the provided list of legal moves."""
    chosen_move = random.choice(legal_moves)
    return chosen_move

def get_winner(info, terminated, truncated):   # this code figures out who won the game or if it is a draw
    """Interpret the environment's termination flag and current agent
    to determine who won, or whether the result was a draw."""
    if terminated:
        if info["agent"] == "player_1":
            return "player_2"
        else:
            return "player_1"

    if truncated:
        return "draw"

    return None


def simulate_random_game(model, player): # rollout : playing an imaginary game
    """Copy the board and finish an imaginary game using random legal moves.
    Returns 1 or 2 for the winning player, or 0 for a draw/cutoff."""

    imaginary_board = model.clone()
    moves_played = 0

    while not imaginary_board.game_over() and moves_played < 200:

        legal_moves = imaginary_board.legal_moves(player)

        if len(legal_moves) == 0:
            return 3 - player

        chosen_move = random.choice(legal_moves)
        imaginary_board.make_move(player, chosen_move)
        player = 3 - player # switches between players

        moves_played = moves_played + 1

    if imaginary_board.get_phase(1) == "lost":
        return 2

    if imaginary_board.get_phase(2) == "lost":
        return 1

    return 0


def choose_monte_carlo_move(model, player): # basic monte carlo search without search tree or uct
    """Try a few random rollouts from each legal move and choose the move
    with the greatest number of simulation wins. This is not tree-based MCTS."""

    legal_moves = model.legal_moves(player)

    best_move = None
    best_wins = -1

    for move in legal_moves:

        test_board = model.clone()
        test_board.make_move(player, move)

        wins = 0

        for simulation_number in range(10):

            winner = simulate_random_game(test_board, 3 - player)

            if winner == player:
                wins = wins + 1

        if wins > best_wins:
            best_wins = wins
            best_move = move

    return best_move





game = mill.env(render_mode=None)

player_1_wins = 0
player_2_wins = 0
draws = 0

for game_number in range(10):

    game.reset()
    turns_played = 0

    while True:

        observation, reward, terminated, truncated, info = game.last()

        if terminated or truncated:

            winner = get_winner(info, terminated, truncated)

            if winner == "player_1":
                player_1_wins = player_1_wins + 1

            elif winner == "player_2":
                player_2_wins = player_2_wins + 1

            else:
                draws = draws + 1

            print("Game", game_number + 1, "winner:", winner)
            break

        model = mill.transition_model(game.unwrapped)

        player = 1 if info["agent"] == "player_1" else 2

        if player == 1:
            chosen_move = choose_monte_carlo_move(model, player)
        else:
            chosen_move = choose_random_move(info["legal_moves"])

        # Only do this during the first move of the first game
        move_for_game = []

        for number in chosen_move:
            move_for_game.append(int(number))

        game.step(move_for_game)

        turns_played = turns_played + 1

print("Player 1 wins:", player_1_wins)
print("Player 2 wins:", player_2_wins)
print("Draws:", draws)

game.close()
