"""Run real matches in FAMNIT Gym without confusing them with MCTS rollouts.

The game environment is the referee. Agent objects merely choose actions.
Importing FAMNIT Gym here (rather than in mcts.py) means search-unit tests
can run independently, even on a computer without FAMNIT installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agents import make_agent
from mcts import normalize_moves


@dataclass
class MatchResult:
    winner: int | None              # 1, 2, 0=draw, None=user quit
    reason: str
    total_turns: int               # Individual moves, not full two-player rounds
    seconds_player1: float         # Total time spent calculating player 1's moves
    seconds_player2: float
    moves_player1: int
    moves_player2: int

    def as_dict(self) -> dict:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "total_turns": self.total_turns,
            "seconds_player1": round(self.seconds_player1, 6),
            "seconds_player2": round(self.seconds_player2, 6),
            "moves_player1": self.moves_player1,
            "moves_player2": self.moves_player2,
        }


def remaining_turns(info: dict, player: int) -> int:
    """Keep imagined games within the real 100-*round* Mill limit.

    FAMNIT's info['move'] starts at 1 and increases AFTER both players move.
    At the start of round 100, player 1 gets two turns remaining (one for
    each player); once player 1 acts, player 2 has only one.
    """
    round_number = int(info.get("move", 1))
    round_number = min(100, max(1, round_number))
    return max(1, 2 * (101 - round_number) - (1 if player == 2 else 0))


def winner_from_model(model: Any) -> int:
    """0 means a draw: neither player has phase 'lost'."""
    if model.get_phase(1) == "lost":
        return 2
    if model.get_phase(2) == "lost":
        return 1
    return 0


def play_match(
    player1: str = "mcts",
    player2: str = "random",
    *,
    iterations1: int = 150,
    iterations2: int = 150,
    exploration1: float = 2 ** 0.5,
    exploration2: float = 2 ** 0.5,
    seed: int = 42,
    verbose: bool = True,
    show_board: bool = False,
) -> MatchResult:
    """Play a complete game between two computer-controlled agents.

    Supported names: random, mcts, easy, medium, hard.
    'show_board' prints the ASCII board after actions (no GUI dependency).
    """
    # This is the ONLY third-party import in our own project: the explicitly
    # permitted FAMNIT Gym package. It may have its own dependencies.
    from famnit_gym.envs import mill

    game = mill.env(render_mode="ansi" if show_board else None)
    players = {
        1: make_agent(player1, seed=seed + 100, iterations=iterations1,
                      exploration=exploration1),
        2: make_agent(player2, seed=seed + 200, iterations=iterations2,
                      exploration=exploration2),
    }
    elapsed = {1: 0.0, 2: 0.0}
    counts = {1: 0, 2: 0}
    reason = "ended"

    try:
        game.reset()
        if verbose:
            print(f"START: player 1 = {player1}; player 2 = {player2}; seed = {seed}")

        # PettingZoo automatically iterates whose turn it is.
        for agent_name in game.agent_iter():
            _, _, terminated, truncated, info = game.last()
            if terminated:
                reason = "win"
                break
            if truncated:
                reason = "100-round truncation or closed window"
                break

            player = 1 if agent_name == "player_1" else 2
            # Always take the transition model from the LIVE current board.
            model = mill.transition_model(game.unwrapped)
            legal = normalize_moves(info["legal_moves"])
            if not legal:
                reason = "no legal moves"
                break

            start = perf_counter()
            action = players[player].choose_move(
                model, player, remaining_plies=remaining_turns(info, player)
            )
            elapsed[player] += perf_counter() - start
            counts[player] += 1

            # Very important: FAMNIT replaces an illegal move with a random
            # move. Detect it here instead of silently corrupting experiments.
            if tuple(action) not in legal:
                raise AssertionError(
                    f"{agent_name} chose ILLEGAL move {action}; legal: {legal}"
                )

            if verbose:
                print(f"  {agent_name} ({player1 if player == 1 else player2}) -> {action}")
            game.step(action)

            if show_board:
                board_text = game.render()
                if board_text is not None:
                    print(board_text)

            # The environment can flag terminal immediately after this move;
            # avoid asking the losing player to make a subsequent action.
            if any(game.terminations.values()):
                reason = "win"
                break
            if any(game.truncations.values()):
                reason = "100-round limit"
                break

        ending_model = mill.transition_model(game.unwrapped)
        winner = winner_from_model(ending_model)
        if winner == 0 and reason == "ended":
            reason = "draw / truncation"
        if verbose:
            print(f"RESULT: {'draw' if winner == 0 else 'player ' + str(winner) + ' won'}")
            print(f"Turns: {sum(counts.values())}; calculation time: {elapsed}")

        return MatchResult(
            winner=winner, reason=reason,
            total_turns=sum(counts.values()),
            seconds_player1=elapsed[1], seconds_player2=elapsed[2],
            moves_player1=counts[1], moves_player2=counts[2],
        )
    finally:
        game.close()
