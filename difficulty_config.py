"""Single source of truth for AI difficulty settings in this project.

EDIT ONLY THIS FILE to change named Easy/Medium/Hard/Insanity profiles.
All project entry points and experiments import their settings from here.

The usual game and iteration-based tournament use ``iterations`` and DEFAULT_C.
Other experimental fields are intentionally separate experimental variables:
- c_study: C differs; each agent gets C_STUDY_ITERATIONS iterations.
- random_move_probability: chance of making a random move instead of MCTS;
  each MCTS call gets RANDOM_MOVE_STUDY_ITERATIONS iterations.
- time_budget_seconds: wall-clock search budget (not a fixed iteration count).

To add a new named level, copy one whole entry with all four fields.
"""

DIFFICULTIES = {
    "easy": {
        "iterations": 100,
        "c_study": 3.0,
        "random_move_probability": 0.35,
        "time_budget_seconds": 0.02,
    },
    "medium": {
        "iterations": 500,
        "c_study": 1.4,
        "random_move_probability": 0.10,
        "time_budget_seconds": 0.09,
    },
    "hard": {
        "iterations": 1000,
        "c_study": 0.3,
        "random_move_probability": 0.00,
        "time_budget_seconds": 0.18,
    },
     "insanity": {
        "iterations": 5000,
        "c_study": 1.4,
         "random_move_probability": 0.00,
         "time_budget_seconds": 0.60,
     },
}

# This C is shared by NORMAL matches, human play and the iteration tournament.
DEFAULT_C = 1.4

# Experiments designed to vary only ONE factor keep their own fixed budgets.
C_STUDY_ITERATIONS = 100
RANDOM_MOVE_STUDY_ITERATIONS = 50

LEVEL_ITERATIONS = {name: values["iterations"] for name, values in DIFFICULTIES.items()}
C_STUDY_VALUES = {name: values["c_study"] for name, values in DIFFICULTIES.items()}
RANDOM_MOVE_PROBABILITIES = {
    name: values["random_move_probability"] for name, values in DIFFICULTIES.items()
}
TIME_BUDGETS = {name: values["time_budget_seconds"] for name, values in DIFFICULTIES.items()}


def validate_config():
    """Fail early on invalid settings rather than executing confusing matches."""
    if len(DIFFICULTIES) < 2:
        raise ValueError("Add at least two difficulty levels")
    for name, values in DIFFICULTIES.items():
        if not isinstance(values["iterations"], int) or values["iterations"] < 1:
            raise ValueError(f"{name}: iterations must be a positive integer")
        if values["c_study"] < 0:
            raise ValueError(f"{name}: c_study must be >= 0")
        if not 0 <= values["random_move_probability"] <= 1:
            raise ValueError(f"{name}: random_move_probability must be between 0 and 1")
        if values["time_budget_seconds"] <= 0:
            raise ValueError(f"{name}: time_budget_seconds must be positive")
    if DEFAULT_C < 0 or C_STUDY_ITERATIONS < 1 or RANDOM_MOVE_STUDY_ITERATIONS < 1:
        raise ValueError("Invalid shared MCTS parameters")


validate_config()
