# FAMNIT Nine Men's Morris — shared difficulty configuration

## Change the difficulty in ONE file

Open **`difficulty_config.py`**. Edit only `iterations` for a normal named difficulty:

```python
DIFFICULTIES = {
    "easy": {"iterations": 100, "c_study": 3.0, "random_move_probability": 0.35, "time_budget_seconds": 0.02},
    "medium": {"iterations": 500, "c_study": 1.4, "random_move_probability": 0.10, "time_budget_seconds": 0.09},
    "hard": {"iterations": 1000, "c_study": 0.3, "random_move_probability": 0.00, "time_budget_seconds": 0.18},
}
```

All the values in this table are **centralized**. If you change `hard` from 1000 to 5000, **`run.py match`**, **`run.py human`**, **`human_vs_mcts.py`**, and **`run.py experiment --study difficulties` / `tournament.py`** will all use 5000 after you restart the command. There are no duplicate iteration dictionaries in these files.

`DEFAULT_C = 1.4` is also in the config. It controls C for *normal matches*, *human play* and *the iteration-based difficulty tournament*. To add `insanity`, uncomment its entire entry in `difficulty_config.py`. The runner automatically accepts the new name and the iteration tournament automatically creates all pairings (more games).

**Important experiment distinction:** The `c-constant`, `random-moves`, and `time-budget` studies intentionally hold the search budget fixed or limit search by time so they can isolate **one independent variable**. Their special C values, random-move probabilities and time budgets are **also in the same config file**, but changing only `iterations` will **not** alter the fixed iteration budgets of those studies. Their fixed budgets appear under `C_STUDY_ITERATIONS` and `RANDOM_MOVE_STUDY_ITERATIONS`. Your MCTS implementation (`expansion_practice.py`) and rollout code (`simulate.py`) need no changes to share the settings; they receive parameters from their callers.

## File names and installation

Use the renamed files from this ZIP, without `(1)` or `(2)` in their filenames. Keep the files together in one directory, otherwise Python imports will fail. The `run.py` script also requires the four original experiment helpers and graph generator, which are included.

Python 3.10+ with Git:

```bash
python -m pip install -r requirements.txt
python run.py config
python run.py smoke
```

`smoke` and game experiments require an installed `famnit_gym` package; Python syntax checks do not.

## Run games and experiments

```bash
# Print the values actually used by all standard difficulty modes:
python run.py config

# Human vs AI with shared Hard setting:
python run.py human --difficulty hard --human-seat 1

# Named-level match:
python run.py match --p1 easy --p2 hard --games-per-seat 2 --output-dir results/easy_vs_hard

# Iteration-based difficulty tournament (shared configuration):
python run.py experiment --study difficulties --games-per-seat 5 --output-dir results/difficulties_updated

# Create graphs from that tournament's CSV files:
python run.py analyze --input-dir results/difficulties_updated --output-dir results/difficulties_updated

# These are deliberately different studies, also centrally configured:
python run.py experiment --study c-constant --games-per-seat 5 --output-dir results/c_constant
python run.py experiment --study random-moves --games-per-seat 5 --output-dir results/random_moves
python run.py experiment --study time-budget --games-per-seat 5 --output-dir results/time_budget
python run.py experiment --study uct-random --games-per-seat 5 --iterations 100 --output-dir results/uct_vs_random
```

**Manual iteration overrides:** `python run.py match --p1 mcts --iterations1 150 ...` uses 150 for the generic `mcts` agent. `--p1 hard` uses the shared `hard` profile instead, regardless of `--iterations1`. The `--iterations` option on `run.py experiment` applies to `uct-random` only.

**Test with just two games per matchup:**

```powershell
python mcts_parameter_experiments/c_study.py --study tournament --games-per-seat 1
python mcts_parameter_experiments/c_study.py --study fixed --games-per-seat 1
python mcts_parameter_experiments/depth_study.py --games-per-seat 1 --iterations 100 --opponent-iterations 100
```

**Poster-ready C × difficulty experiment (4 × 3 × 30 = 360 games):**

```powershell
python mcts_parameter_experiments/c_study.py --study tournament --c-values 0 0.3 1.4 3 --easy 10 --medium 50 --hard 100 --games-per-seat 15 --output-dir results/c_difficulties
```

**Isolate C against unchanged opponent (6 × 30 = 180 games):**

```powershell
python mcts_parameter_experiments/c_study.py --study fixed --c-values 0 0.3 0.7 1.4 2 3 --iterations 100 --opponent-iterations 100 --opponent-c 1.4 --games-per-seat 15 --output-dir results/c_fixed
```
# Quick commands

Edit `difficulty_config.py` to change Easy/Medium/Hard/Insanity.

```bash
python run.py config
python run.py smoke
python run.py human --difficulty hard --human-seat 1
python run.py match --p1 easy --p2 hard --games-per-seat 2
python run.py experiment --study difficulties --games-per-seat 5 --output-dir results/difficulties_updated
python run.py analyze --input-dir results/difficulties_updated --output-dir results/difficulties_updated
```

The C-constant, time-budget and random-moves studies are deliberately separate types of difficulty experiments and have their own settings in `difficulty_config.py`.
