"""Beginner practice file: manually create an MCTS root and expand one move.
Useful for seeing the dictionary fields; not the full AI implementation.

Notes: This file is part of the FAMNIT Mill seminar project.
"""

from famnit_gym.envs import mill

game = mill.env(render_mode=None)
game.reset()

model = mill.transition_model(game.unwrapped)

root = {
    "state": model.clone(),
    "to_move": 1,
    "move": None,
    "visits": 0,
    "reward_sum": 0,
    "children": [],
    "untried_moves": model.legal_moves(1)
}
move = root["untried_moves"].pop(0)

new_board = root["state"].clone()

new_board.make_move(1, move)
child = {
    "state": new_board,
    "to_move": 2,
    "move": move,
    "visits": 0,
    "reward_sum": 0,
    "children": [],
    "untried_moves": new_board.legal_moves(2)
}

root["children"].append(child)
print("Root board:", root["state"].get_state())

print("Child board:", child["state"].get_state())

print("Move:", child["move"])

print("Number of children:", len(root["children"]))

game.close()
