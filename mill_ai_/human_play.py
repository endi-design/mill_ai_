"""Human vs MCTS, using the REQUIRED FAMNIT UserInteraction wrapper.

How mouse input is interpreted:
  Placing phase: click destination, then capture if a mill was formed.
  Moving/flying: click source, destination, then capture if required.
  Escape key: restart selection of the current action.
  Close the window: quit the game.

All mouse clicks are restricted to choices that appear in the environment's
legal_moves list. That way, the wrapper input cannot create an illegal move.
"""

from __future__ import annotations

from agents import make_agent
from game import remaining_turns, winner_from_model
from mcts import normalize_moves


class HumanQuit(Exception):
    """Raised when the user closes the window."""


# RGBA highlight colors for selectable positions and the last selection.
AVAILABLE = (30, 170, 60, 130)
SELECTED = (255, 180, 10, 155)


def wait_for_position(ui, options: set[int], prompt: str) -> int | None:
    """Ask for one mouse click on a listed position.

    Return None on Escape (restart action). Other keys or wrong clicks are
    ignored. 'interact()' comes from FAMNIT's UserInteraction wrapper.
    """
    ui.clear_markings()
    for position in sorted(options):
        ui.mark_position(int(position), AVAILABLE)
    print(prompt, "| highlighted positions:", sorted(options))

    while True:
        msg = ui.interact()
        if msg["type"] == "quit":
            raise HumanQuit
        if msg["type"] == "key_press":
            if str(msg.get("key", "")).lower() in ("escape", "esc"):
                return None
        if msg["type"] == "mouse_click":
            position = int(msg.get("position", 0))
            if position in options:
                return position


def choose_human_action(ui, legal: list[tuple[int, int, int]]) -> list[int]:
    """Guide the user through [src, dst, capture], using legal-move filters.

    We never need to implement movement or capture rules ourselves: selecting
    source/destination/capture progressively filters the legal action list.
    """
    while True:
        src_options = {m[0] for m in legal}
        if src_options == {0}:
            source = 0  # Placement: the player has no piece to select.
        else:
            source = wait_for_position(ui, src_options - {0},
                                       "Select the piece you want to move")
            if source is None:
                continue

        matching_src = [m for m in legal if m[0] == source]
        destinations = {m[1] for m in matching_src}
        destination = wait_for_position(ui, destinations,
                                        "Select the destination position")
        if destination is None:
            continue

        matching_dst = [m for m in matching_src if m[1] == destination]
        captures = {m[2] for m in matching_dst}
        if captures == {0}:
            capture = 0
        else:
            capture = wait_for_position(ui, captures - {0},
                                         "Mill formed! Select a piece to capture")
            if capture is None:
                continue

        action = (source, destination, capture)
        if action in legal:
            ui.clear_markings()
            return list(action)
        # Defensive, should not happen because options came from legal moves.
        print("Move was not legal. Try again.")


def play_human(difficulty: str = "medium", human_seat: int = 1,
               seed: int = 42) -> None:
    """Open graphical Mill and let a human play against an MCTS AI."""
    from famnit_gym.envs import mill
    from famnit_gym.wrappers.mill import UserInteraction

    if human_seat not in (1, 2):
        raise ValueError("human_seat must be 1 or 2")

    # Required interaction wrapper; requires the graphical 'human' renderer.
    ui = UserInteraction(mill.env(render_mode="human"))
    ai = make_agent(difficulty, seed=seed)

    try:
        ui.reset()
        ui.render()
        print("Human vs", difficulty)
        print("Click highlighted positions. Press Esc to restart a move.")
        for agent_name in ui.agent_iter():
            _, _, terminated, truncated, info = ui.last()
            if terminated or truncated:
                break
            player = 1 if agent_name == "player_1" else 2
            legal = normalize_moves(info["legal_moves"])
            if not legal:
                break

            if player == human_seat:
                try:
                    action = choose_human_action(ui, legal)
                except HumanQuit:
                    print("Window closed; match cancelled.")
                    return
            else:
                model = mill.transition_model(ui.unwrapped)
                print(f"AI (player {player}) is thinking...")
                action = ai.choose_move(model, player, remaining_turns(info, player))
                print("AI plays:", action)

            # Do not silently accept illegal moves: the env would substitute
            # a random move, potentially hiding a logic bug.
            if tuple(action) not in legal:
                raise AssertionError(f"Illegal game action: {action}")
            ui.step(action)
            if any(ui.terminations.values()) or any(ui.truncations.values()):
                break

        result = winner_from_model(mill.transition_model(ui.unwrapped))
        print("Final result:", "draw" if result == 0 else f"player {result} won")
    finally:
        ui.close()
