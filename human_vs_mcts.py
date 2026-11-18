"""Play against your MCTS agent using the FAMNIT Mill graphical board.

Use `python run.py human --difficulty medium --human-seat 1` to start.
Select highlighted pieces/positions with the mouse; Escape resets a move.
"""

from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
from expansion_practice import choose_mcts_move
from difficulty_config import LEVEL_ITERATIONS as DIFFICULTIES, DEFAULT_C


def get_remaining_moves(info, player):
    """Find how many individual moves remain before the 100-round cutoff."""
    round_number = max(1, min(100, int(info["move"])))
    remaining = (101 - round_number) * 2
    if player == 2:
        remaining -= 1
    return max(0, remaining)


def human_choose_move(game, legal_moves):
    """Collect legal mouse clicks until source, destination and capture are set.

    `legal_moves` is the source of truth: invalid clicks reset selection, and
    only a complete move from this list can ever be returned to the game.
    """
    legal_moves = [[int(x) for x in move] for move in legal_moves]
    selected = []

    while True:
        # Build the list of legal positions for the *next* part of the move.
        next_positions = []
        for move in legal_moves:
            # 0 means 'no source' in the placing phase or 'no capture'.
            positions = [pos for pos in move if pos != 0]
            if positions[:len(selected)] == selected and len(positions) > len(selected):
                next_position = positions[len(selected)]
                if next_position not in next_positions:
                    next_positions.append(next_position)

        # Green squares show which positions can legally be clicked next.
        game.clear_markings()
        for position in next_positions:
            game.mark_position(position, (0, 255, 0, 130))

        message = game.interact()  # Wait for one mouse or keyboard action.
        if message["type"] == "quit":
            return None
        if message["type"] == "key_press":
            if message["key"] in ("escape", "esc"):
                selected = []  # Start choosing the move from scratch.
            continue
        if message["type"] != "mouse_click":
            continue

        position = message["position"]
        if position not in next_positions:
            selected = []
            continue

        selected.append(position)
        # A complete valid move might include a capture (third position).
        for move in legal_moves:
            if [pos for pos in move if pos != 0] == selected:
                game.clear_markings()
                return move


def main(difficulty="medium", human_seat=1):
    """Create the graphical game and alternate human and AI turns until game over."""
    if difficulty not in DIFFICULTIES:
        raise ValueError("Difficulty must be one of: " + ", ".join(DIFFICULTIES))
    if human_seat not in (1, 2):
        raise ValueError("Human seat must be 1 or 2")

    game = UserInteraction(mill.env(render_mode="human"))
    game.reset()
    print("Human vs MCTS")
    print("You are Player", human_seat)
    print("AI difficulty:", difficulty, "—", DIFFICULTIES[difficulty], "iterations; C =", DEFAULT_C)
    print("Click highlighted positions. Escape restarts move selection.")

    try:
        while True:
            _, _, terminated, truncated, info = game.last()
            if terminated:

                final_board = mill.transition_model(game.unwrapped)

                if final_board.get_phase(human_seat) == "lost":
                    print("MCTS wins!")

                elif final_board.get_phase(3 - human_seat) == "lost":
                    print("Human wins!")

                else:
                    print("Game ended, but no losing player was identified.")

                break
            if truncated:
                print("Draw or game window closed.")
                break

            player = 1 if info["agent"] == "player_1" else 2
            if player == human_seat:
                print("Your turn!")
                chosen = human_choose_move(game, info["legal_moves"])
                if chosen is None:
                    print("Game closed.")
                    break
            else:
                print("AI is thinking...")
                model = mill.transition_model(game.unwrapped)
                chosen = choose_mcts_move(model, player, DIFFICULTIES[difficulty],
                                          DEFAULT_C, get_remaining_moves(info, player))
                print("AI played:", [int(x) for x in chosen])

            # FAMNIT converts illegal actions to random legal moves; reject them.
            selected = [int(x) for x in chosen]
            legal = [[int(x) for x in move] for move in info["legal_moves"]]
            if selected not in legal:
                raise ValueError(f"Illegal human or AI action: {selected}")
            game.step(selected)
    finally:
        game.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play against the MCTS agent")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="medium")
    parser.add_argument("--human-seat", type=int, choices=[1, 2], default=1)
    settings = parser.parse_args()
    main(difficulty=settings.difficulty, human_seat=settings.human_seat)
