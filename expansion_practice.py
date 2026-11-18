"""Core MCTS implementation. Defines UCT, selection, expansion,
simulation, backpropagation, and choosing the most-visited move.
"""

from famnit_gym.envs import mill
import random
from simulate import simulate_random_game
import math

def calculate_uct(node, log_parent_visits, C):
    """Compute one child's UCT selection score.
    An unvisited child scores infinity so it gets explored first.
    Reward estimates are from the viewpoint of the player who just moved."""

    if node["visits"] == 0:
        return math.inf

    # Exploitation: how good previous simulations looked for this move.
    average = node["reward_sum"] / node["visits"]

    # Exploration: reward less-visited branches, scaled by C.
    exploration = C * math.sqrt(
        log_parent_visits / node["visits"]
    )

    return average + exploration


def select_best_child(node, C):
    """Find the child with the highest UCT score.
    Selection compares exploration (new moves) against exploitation (good results)."""

    best_child = None
    best_score = -math.inf

    # Calculate the logarithm only once
    log_parent_visits = math.log(node["visits"])

    for child in node["children"]:

        score = calculate_uct(
            child,
            log_parent_visits,
            C
        )

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def expand_node(node):
    """Choose a previously untried legal move, clone the board, and create
    a child node for the resulting imaginary position."""

    # Choose a RANDOM unexplored move; avoid always preferring early list items
    index = random.randrange(len(node["untried_moves"]))

    move = node["untried_moves"].pop(index)

    # Copy the board
    new_board = node["state"].clone()

    # Find whose turn it is
    player = node["to_move"]

    # Perform the imaginary move
    new_board.make_move(player, move)

    # Switch players
    next_player = 3 - player
    new_depth = node["depth"] + 1
    # Create the new child
    child = {
        "state": new_board,
        "to_move": next_player,
        "move": move,
        "visits": 0,
        "reward_sum": 0,
        "depth": new_depth,
        "children": [],
        "untried_moves": new_board.legal_moves(next_player),
        "parent": node,
        "just_moved": player
    }

    # Connect child to parent
    node["children"].append(child)

    return child

def backpropagate(node, winner):
    """Walk from the simulated node up to the root, adding one visit to
    each node and recording the simulation outcome for the last mover."""

    while node is not None:

        node["visits"] = node["visits"] + 1

        if winner != 0:

            if winner == node["just_moved"]:
                node["reward_sum"] += 1

            else:
                node["reward_sum"] -= 1

        node = node["parent"]



def mcts_iteration(root, C, remaining_moves):
    """Run one complete MCTS iteration: selection, expansion, random rollout,
    and backpropagation. Respect the remaining legal move budget."""

    node = root

    # STEP 1: SELECTION
    while (
        node["depth"] < remaining_moves
        and not node["state"].game_over()
        and len(node["untried_moves"]) == 0
        and len(node["children"]) > 0
    ):
        node = select_best_child(node, C)

    # STEP 2: EXPANSION
    if (
        node["depth"] < remaining_moves
        and not node["state"].game_over()
        and len(node["untried_moves"]) > 0
    ):
        node = expand_node(node)

    # STEP 3: SIMULATION. The tree already used node["depth"] moves.
    # Do not let the rollout exceed the real game's move budget.
    max_moves = remaining_moves - node["depth"]

    winner = simulate_random_game(
        node["state"],
        node["to_move"],
        max_moves
    )

    # STEP 4: BACKPROPAGATION (update visits/reward up to root)
    backpropagate(node, winner)

    return winner



def choose_best_move(root):
    """Choose the action from the root's most-visited child.
    This is the actual move returned to the environment after search."""

    best_move = None
    most_visits = -1

    for child in root["children"]:

        if child["visits"] > most_visits:
            most_visits = child["visits"]
            best_move = child["move"]

    return best_move


def choose_mcts_move(model, player, iterations, C, remaining_moves):
    """Build a new MCTS search tree and repeat it a fixed number of times.
    Return a legal move for the current player."""
    legal_moves = model.legal_moves(player)

    if len(legal_moves) == 1:
        return legal_moves[0]
    # Create the starting node
    root = {
        "state": model.clone(),
        "to_move": player,
        "just_moved": 3 - player,
        "move": None,
        "parent": None,
        "depth": 0,
        "visits": 0,
        "reward_sum": 0,
        "children": [],
        "untried_moves": model.legal_moves(player)
    }

    # Run the MCTS search
    for i in range(iterations):
        mcts_iteration(root, C, remaining_moves)

    # Choose the move with the most visits
    best_move = choose_best_move(root)

    return best_move

if __name__ == "__main__":

    game = mill.env(render_mode=None)
    game.reset()

    model = mill.transition_model(game.unwrapped)

    root = {
        "state": model.clone(),
        "to_move": 1,
        "just_moved": 2,
        "move": None,
        "parent": None,
        "depth": 0,
        "visits": 0,
        "reward_sum": 0,
        "children": [],
        "untried_moves": model.legal_moves(1)
    }

    child = expand_node(root)

    grandchild = expand_node(child)

    print("Root depth:", root["depth"])
    print("Child depth:", child["depth"])
    print("Grandchild depth:", grandchild["depth"])

    game.close()
