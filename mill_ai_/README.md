# Nine Men's Morris (Mill): Intelligent Systems seminar

**Complete commented example project** for studying and implementing Monte Carlo Tree Search (MCTS) in the UP FAMNIT Mill game.

This project covers:

1. Adversarial search: MCTS, MCTS vs random, MCTS vs MCTS, UCT comparisons.
2. Different difficulties: easy / medium / hard using different simulation budgets and optional mistakes, difficulty comparisons and timing.
3. Human vs AI: graphical FAMNIT `UserInteraction` wrapper.
4. Source code and experiments: commented Python modules, source CSVs, and standard-library SVG plots.

**Important academic note:** The assignment says all algorithms must be implemented by students. Use this project as an educational reference, understand and adapt the design, test it yourself, and follow your course's rules on outside assistance. Do not present unverified or invented experiment results as measured outcomes.

## 1. Python requirements

Python **3.10+** (3.11 or 3.12 recommended), Git, and the FAMNIT Gym package. This project imports **no third-party libraries beyond `famnit_gym`**. Its own code uses Python's standard library (`random`, `math`, `csv`, `argparse`, `unittest`, etc.). Dependencies required by `famnit_gym` itself are installed automatically by pip.

### Windows / VS Code / PowerShell

Open this project folder in VS Code; use **Terminal > New Terminal**.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "famnit_gym @ git+https://github.com/DomenSoberlFamnit/famnit-gym.git"
.\.venv\Scripts\python.exe smoke_test.py
```

If your computer uses a different Python version, replace `py -3.12` with `py` (or a supported Python launcher). If `pip` reports **Git not found**, install Git for Windows and restart the terminal.

All further commands below use `python` for readability. In PowerShell, either activate your venv or substitute `.\.venv\Scripts\python.exe` for `python`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "famnit_gym @ git+https://github.com/DomenSoberlFamnit/famnit-gym.git"
python smoke_test.py
```

## 2. Folder and file descriptions

```text
Mill_Seminar_Commented_Code/
    mcts.py                # MCTS tree + UCT + four search phases
    agents.py              # Random baseline + difficulty settings
    game.py                # Turn-based real matches and move validation
    human_play.py          # Human vs AI using UserInteraction (mouse)
    experiments.py         # Balanced experiments, CSV and SVG plots
    run.py                 # Command-line entry point
    smoke_test.py          # Quick check against the real FAMNIT game
    tests/test_mcts.py     # Stand-alone search tests
    tests/test_experiments.py # CSV/SVG experiment test
    README.md              # This guide
    RUN_INSTRUCTIONS.md    # Concise commands for submission
```

## 3. Play matches (Problem 1)

MCTS versus a random opponent:

```bash
python run.py match --p1 mcts --p2 random --iterations1 100
```

Swap seats:

```bash
python run.py match --p1 random --p2 mcts --iterations2 100
```

MCTS versus MCTS with different exploration coefficients:

```bash
python run.py match --p1 mcts --p2 mcts --iterations1 150 --iterations2 150 --c1 0.5 --c2 2.0
```

Print an ASCII version of the board after each move:

```bash
python run.py match --p1 mcts --p2 random --iterations1 30 --show-board
```

The default MCTS value `C` is `sqrt(2) ≈ 1.414`.

## 4. Experiment with UCT (Problem 1)

Quick functionality check (small sample, not enough for conclusions):

```bash
python run.py experiment --study uct-random --games-per-seat 1 --iterations 30 --output-dir results/uct_quick
```

More meaningful experiment (may take substantially longer):

```bash
python run.py experiment --study uct-random --games-per-seat 10 --iterations 150 --output-dir results/uct_random
```

Compare UCT values directly against MCTS with `C = sqrt(2)`:

```bash
python run.py experiment --study uct-mcts --games-per-seat 10 --iterations 150 --baseline-iterations 150 --output-dir results/uct_mcts
```

Choose your own coefficients:

```bash
python run.py experiment --study uct-random --c-values 0 0.25 0.5 1 1.414 2 3 --games-per-seat 5 --iterations 100
```

Every comparison uses each seat (`player_1` and `player_2`) the same number of times. Seeds are recorded. Results go to:

- `games.csv`: each individual match, outcome, seed, decisions made, timings;
- `summary.csv`: wins, draws, losses, **win rate** and **score rate**;
- `score_rate.svg`: graph of points scored, counting a draw as 0.5;
- `time_per_move.svg`: mean decision time graph.

**Use actual CSV values when preparing your poster**. Small numbers of games can be misleading, and running the same random seeds and program version does not guarantee identical CPU timings across computers.

## 5. Three difficulties (Problem 2)

Settings in `agents.py`:

| Level | MCTS iterations per move | Chance of random move |
|-------|-------------------------:|----------------------:|
| easy | 30 | 12% |
| medium | 120 | 2% |
| hard | 400 | 0% |

Play levels against each other:

```bash
python run.py match --p1 easy --p2 hard
python run.py match --p1 medium --p2 medium
```

Run all three pairwise comparisons, with alternating seats:

```bash
python run.py experiment --study difficulties --games-per-seat 8 --output-dir results/difficulties
```

The output gives average think time per move, which you can compare between levels. **More iterations increase computation, but don't mathematically guarantee a win**; random rollouts and game variance still matter.

## 6. Play as a human (Problem 3)

The human mode requires a graphical desktop (not a terminal-only / headless server):

```bash
python run.py human --difficulty easy --human-seat 1
python run.py human --difficulty hard --human-seat 2
```

On your turn, click the green-highlighted selectable position(s):

- **Placing:** click the destination position.
- **Moving or flying:** click the piece you want to move, then its destination.
- **Closing a mill:** click the opponent's highlighted piece that you want to capture.
- **Esc:** restart selecting the current move.
- **Close the window:** end the match.

The interface uses FAMNIT's `UserInteraction` wrapper (`ui.interact`, `ui.mark_position`, `ui.clear_markings`). If an input is not a legal move, no action is submitted.

Note: since we cannot run the GUI in this environment, test mouse interactions locally; minor wrapper/event differences between FAMNIT versions may require adjustment.

## 7. Tests

These tests need only the standard library and can run without FAMNIT Gym installed:

```bash
python -m unittest discover -s tests -v
```

Check the actual FAMNIT integration once installed:

```bash
python smoke_test.py
```

Do this before collecting seminar results.

## 8. How MCTS works (explain this in your presentation)

One **tree node** represents a cloned game position and stores:

- which player's turn comes next;
- its move from the parent, parent link, children and untried actions;
- number of visits and accumulated rewards.

Each iteration performs:

1. **Selection:** follow high-UCT children while all actions in the current node were expanded.
2. **Expansion:** pick one untried legal move; clone and apply that move to create a child.
3. **Simulation (rollout):** complete a hypothetical game using random legal moves from both players.
4. **Backpropagation:** update each node along the chosen path with +1 for a win, -1 for a loss, or 0 for a draw from the appropriate player's viewpoint.

The UCT formula is:

```text
UCT(child) = child.mean_reward + C * sqrt(ln(parent.visits) / child.visits)
```

It balances **exploitation** (known promising moves) with **exploration** (less-tested moves). Each child measures rewards from the perspective of its **incoming mover**, making the selection adversarial for both players. At the end of the allotted iterations, the most-visited root child supplies the actual move.

## 9. Accuracy and honest limitations

- Code compiles and self-contained unit tests can run without the actual FAMNIT environment.
- The official FAMNIT package cannot be installed here because this runtime does not have outbound GitHub networking. **A fully verified FAMNIT match and GUI have NOT been run here**; run `smoke_test.py` and mouse tests on your own computer.
- Mill's model uses `game_over()` for a lost player; the environment also has a **100-round truncation**. We therefore stop hypothetical games at the remaining number of individual player turns computed from `info['move']`.
- The random-rollout policy is deliberately simple. You may improve it with a smarter rollout strategy, but test the baseline before changing it.
- The GUI needs a graphical desktop and a compatible FAMNIT installation.

## 10. Preparing the course submission

The course requires **three separate files**, named like:

```text
<group>_IS_Seminar1_<surname1>_<surname2>.zip
<group>_IS_Seminar1_<surname1>_<surname2>.pdf
<group>_IS_Seminar1_<surname1>_<surname2>.md
```

- ZIP: this source project, renamed appropriately after you verify it on your PC.
- PDF: your conference-style poster containing **actual measured results** (not generated by this code yet).
- MD: use `RUN_INSTRUCTIONS.md`, renamed to the required format.

**Both group members must submit the same files** in their own accounts, and the submission is due one day before the presentation. Test precisely the code you submit and rerun the experimental scripts to verify poster figures.

### Repository sources

- https://github.com/DomenSoberlFamnit/famnit-gym
- https://github.com/DomenSoberlFamnit/famnit-gym/blob/main/docs/mill.md
