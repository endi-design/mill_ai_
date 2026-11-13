"""Monte Carlo Tree Search (MCTS) for the FAMNIT Nine Men's Morris game.

This file uses only Python's standard library. The FAMNIT transition model
implements game rules; we implement ALL decision-making/search logic here.

Required transition-model methods:
    clone(), legal_moves(player), make_move(player, move),
    game_over(), get_phase(player)

Important: in this implementation, each non-root node stores its average
reward from the perspective of the player who MADE its incoming move. This
means both players can maximize UCT while adversarially choosing moves.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Any


# A Mill move is [source, destination, capture]. We use tuples inside the tree
# so they cannot accidentally be modified, then return a list to the game.
Move = tuple[int, int, int]

#Choses which player you will be
def other_player(player: int) -> int:
    """Switch 1 <-> 2. (3 - 1 = 2 and 3 - 2 = 1.)"""
    if player not in (1, 2):
        raise ValueError("Player must be 1 or 2")
    return 3 - player


def normalize_moves(raw_moves: Any) -> list[Move]:
    """Convert FAMNIT's NumPy array of legal moves to plain Python tuples.

    NumPy is used internally by FAMNIT Gym, but this project does not need to
    import it: iterating the array and converting numbers with int() is enough.
    """
    return [tuple(int(x) for x in move) for move in raw_moves]


def evaluate_terminal(state: Any, perspective: int):
    """Return +1 for our win, -1 for our loss, or 0 for a draw/cutoff.

    The Mill transition model sets get_phase(player) to 'lost' when that
    player has lost. It does NOT itself enforce the environment's 100-round
    limit, so a playout that hits our cutoff is scored as a draw.
    """
    if state.get_phase(perspective) == "lost":
        return -1.0
    if state.get_phase(other_player(perspective)) == "lost":
        return 1.0
    return 0.0


@dataclass
class Node:
    """A node holds one board position and the statistics for that position.

    A node corresponds to the position AFTER its incoming 'move'.
    'to_move' is the player who will choose the next action from this board.
    'just_moved' is the one who made the incoming action.
    """

    state: Any
    to_move: int
    just_moved: int
    depth: int
    ply_limit: int
    move: Move | None = None
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    untried_moves: list[Move] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0

    def __post_init__(self) -> None:
        # Compute the legal actions once when this position enters the tree.
        # Terminal positions have no actions to explore.
        if not self.is_terminal():
            self.untried_moves = normalize_moves(self.state.legal_moves(self.to_move))

    def is_terminal(self) -> bool:
        """Game ended, or we've reached the safety/correctness move limit."""
        return self.depth >= self.ply_limit or bool(self.state.game_over())

    def average_reward(self) -> float:
        """Mean outcome from the perspective of the previous mover."""
        if self.visits == 0:
            return 0.0
        return self.reward_sum / self.visits

    def uct_score(self, exploration: float) -> float:
        """UCT = mean reward + C * sqrt(ln(parent visits) / visits).

        The first term exploits positions that worked well in prior rollouts.
        The second term explores positions we have tested relatively little.
        An unvisited child is given +infinity so it gets explored first.
        """
        if self.visits == 0:
            return math.inf
        if self.parent is None:
            raise ValueError("UCT is defined for a child with a parent")
        exploitation = self.average_reward()
        exploration_bonus = exploration * math.sqrt(
            math.log(max(1, self.parent.visits)) / self.visits
        )
        return exploitation + exploration_bonus


class MCTSAgent:
    """AI agent using four MCTS phases and UCT for tree selection.

    iterations: number of simulated games per real move. More is usually
        stronger but slower; useful for easy/medium/hard difficulty levels.
    exploration: UCT constant C. 0 = greedy exploitation, larger = explore.
    mistake_rate: chance to deliberately select a random legal action. This
        is optional and helps create an easier, beatable opponent.
    seed: used for repeatable experiments (one random generator per agent).
    """

    def __init__(
        self,
        iterations: int = 150,
        exploration: float = math.sqrt(2),
        mistake_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be a finite, nonnegative number")
        if not 0 <= mistake_rate <= 1:
            raise ValueError("mistake_rate must be in [0, 1]")

        self.iterations = iterations
        self.exploration = exploration
        self.mistake_rate = mistake_rate
        self.rng = random.Random(seed)
        self.last_root_statistics: list[dict[str, object]] = []

    def rollout(
        self,
        state: Any,
        next_player: int,
        start_depth: int,
        root_player: int,
        ply_limit: int,
    ) -> float:
        """PHASE 3: randomly finish an imaginary game from a cloned state.

        'state' is owned by the rollout; it is safe to modify it. By using
        random moves here, we avoid needing a hand-written chess-like
        evaluation function. The game outcome teaches the tree which actions
        tend to lead to wins and losses.
        """
        depth = start_depth
        while depth < ply_limit and not state.game_over():
            moves = normalize_moves(state.legal_moves(next_player))
            if not moves:
                # Defensive fallback: in Mill, a player with no legal moves
                # normally already has phase 'lost'.
                return -1.0 if next_player == root_player else 1.0

            chosen = self.rng.choice(moves)
            state.make_move(next_player, list(chosen))
            next_player = other_player(next_player)
            depth += 1

        return evaluate_terminal(state, root_player)

    def choose_move(self, model: Any, player: int, remaining_plies: int = 200) -> list[int]:
        """Return a legal action [src, dst, capture] for the live game.

        model: current FAMNIT transition model (NOT the live env itself).
        player: 1 or 2, must be the player currently taking a turn.
        remaining_plies: max additional individual turns before a draw; use
            the game's 'move' counter to respect its 100-round truncation.
        """
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        if remaining_plies < 1:
            raise ValueError("remaining_plies must be >= 1")
        if model.game_over():
            raise ValueError("cannot play from an ended position")

        legal = normalize_moves(model.legal_moves(player))
        if not legal:
            raise ValueError("no legal moves available")

        # Optional deliberate error, mainly for the 'easy' difficulty setting.
        if self.mistake_rate and self.rng.random() < self.mistake_rate:
            action = self.rng.choice(legal)
            self.last_root_statistics = []
            return list(action)

        # Only the copy is used for the tree: the actual board stays untouched.
        root = Node(
            state=model.clone(),
            to_move=player,
            just_moved=other_player(player),
            depth=0,
            ply_limit=remaining_plies,
        )

        for _ in range(self.iterations):
            node = root

            # ----- PHASE 1: SELECTION -----
            # Follow already-created nodes while this position has no
            # unexplored moves. Each child measures rewards from the POV of
            # the player who selected it, so the opponent also plays to WIN.
            while (not node.is_terminal() and
                   not node.untried_moves and
                   node.children):
                highest = max(ch.uct_score(self.exploration) for ch in node.children)
                tied = [ch for ch in node.children
                        if ch.uct_score(self.exploration) == highest]
                node = self.rng.choice(tied)

            # ----- PHASE 2: EXPANSION -----
            # Create exactly one child for an unexplored legal action.
            # The action modifies a clone, not the parent's board.
            if not node.is_terminal() and node.untried_moves:
                index = self.rng.randrange(len(node.untried_moves))
                action = node.untried_moves.pop(index)
                next_state = node.state.clone()
                next_state.make_move(node.to_move, list(action))
                child = Node(
                    state=next_state,
                    to_move=other_player(node.to_move),
                    just_moved=node.to_move,
                    depth=node.depth + 1,
                    ply_limit=remaining_plies,
                    move=action,
                    parent=node,
                )
                node.children.append(child)
                node = child

            # ----- PHASE 3: SIMULATION -----
            # Evaluate one possible future by completing a random game.
            reward = self.rollout(
                state=node.state.clone(),
                next_player=node.to_move,
                start_depth=node.depth,
                root_player=player,
                ply_limit=remaining_plies,
            )

            # ----- PHASE 4: BACKPROPAGATION -----
            # Send the outcome from the selected node all the way to root.
            # A +1 outcome for the root player is a -1 outcome for its rival.
            while node is not None:
                node.visits += 1
                if node.just_moved == player:
                    node.reward_sum += reward
                else:
                    node.reward_sum -= reward
                node = node.parent

        # At decision time, choose the MOST VISITED child, not max UCT.
        # UCT includes exploration bonuses designed for searching, not for
        # picking the final actual move.
        max_visits = max(child.visits for child in root.children)
        most_visited = [ch for ch in root.children if ch.visits == max_visits]
        best = self.rng.choice(most_visited)

        # Diagnostics useful for your poster and for debugging decisions.
        self.last_root_statistics = sorted(
            [
                {
                    "move": list(child.move),
                    "visits": child.visits,
                    "average_reward": round(child.average_reward(), 4),
                }
                for child in root.children
            ],
            key=lambda row: row["visits"],
            reverse=True,
        )

        # Catch illegal actions before the environment can silently replace
        # them with a random action (a documented FAMNIT behavior).
        if best.move not in legal:
            raise AssertionError("MCTS selected an illegal move")
        return list(best.move)
