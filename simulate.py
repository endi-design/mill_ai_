"""Random rollout helper for MCTS. Copies the board, plays the best/random legal
moves until the game ends or the requested number of moves has elapsed.
"""

import random
def simulate_random_game(model, player, max_moves=200):
    """Copy the board and finish an imaginary game using the best looking legal moves or a random move.
    Returns 1 or 2 for the winning player, or 0 for a draw"""
    imaginary_board = model.clone()
    moves_played = 0

    while not imaginary_board.game_over() and moves_played < max_moves:
        legal_moves = imaginary_board.legal_moves(player)

        if len(legal_moves) == 0:
            return 3 - player

        # Find moves that form a mill and capture a piece
        capture_moves = []

        for move in legal_moves:
            if int(move[2]) != 0:
                capture_moves.append(move)

        # Prefer capturing moves 90% of the time
        if capture_moves and random.random() < 0.9:
            chosen_move = random.choice(capture_moves)
        else:
            chosen_move = random.choice(legal_moves)
        imaginary_board.make_move(player, chosen_move)
        player = 3 - player # switches between players

        moves_played = moves_played + 1

    if imaginary_board.get_phase(1) == "lost":
        return 2

    if imaginary_board.get_phase(2) == "lost":
        return 1

    return 0
