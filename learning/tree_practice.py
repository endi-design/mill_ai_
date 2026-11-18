"""Learning exercise: two toy moves are scored with UCT to explain
how the search selects which child node to investigate next.

Notes: this file is just for practice we don't need it in project
"""

import math

def calculate_uct(node, parent_visits, C):
    """Compute one child's UCT selection score.
    An unvisited child scores infinity so it gets explored first.
    Reward estimates are from the viewpoint of the player who just moved."""

    if node["visits"] == 0:
        return math.inf

    average = node["reward_sum"] / node["visits"]

    exploration = C * math.sqrt(
        math.log(parent_visits) / node["visits"]
    )

    uct = average + exploration

    return uct

root = {
    "move": None,
    "visits": 12,
    "reward_sum": 0,
    "children": []
}

move_a = {
    "move": [0, 1, 0],
    "visits": 10,
    "reward_sum": 8,
    "children": []
}

move_b = {
    "move": [0, 2, 0],
    "visits": 2,
    "reward_sum": 0,
    "children": []
}

root["children"].append(move_a)
root["children"].append(move_b)
print(root["children"])

for child in root["children"]:

    score = calculate_uct(child, root["visits"], 1.4)

    print("Move:", child["move"])
    print("UCT:", score)

best_child = None
best_score = -math.inf

for child in root["children"]:

    score = calculate_uct(child, root["visits"], 1.4)

    if score > best_score:
        best_child = child
        best_score = score

print("Best move:", best_child["move"])
print("Best UCT score:", best_score)


def select_best_child(root, C):
    """Find the child with the highest UCT score.
    Selection compares exploration (new moves) against exploitation (good results)."""

    best_child = None
    best_score = -math.inf

    for child in root["children"]:

        score = calculate_uct(child, root["visits"], C)

        if score > best_score:
            best_child = child
            best_score = score
            # Save the best child here

            # Save its score here

    return best_child

selected = select_best_child(root, 1.4)

print("Selected move:", selected["move"])
