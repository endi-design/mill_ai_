# Running the Nine Men's Morris MCTS seminar code

Requires Python 3.10+, Git, and FAMNIT Gym. Open a terminal in the code folder.

Install FAMNIT Gym:

```bash
python -m pip install "famnit_gym @ git+https://github.com/DomenSoberlFamnit/famnit-gym.git"
```

Windows PowerShell users can replace `python` by `py` or the Python interpreter inside their virtual environment.

Check the installation and source tests:

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

Run AI vs random, and AI vs AI:

```bash
python run.py match --p1 mcts --p2 random --iterations1 100
python run.py match --p1 mcts --p2 mcts --c1 0.5 --c2 2.0
```

Test different UCT values and save CSV + SVG data:

```bash
python run.py experiment --study uct-random --games-per-seat 10 --iterations 150 --output-dir results/uct_random
python run.py experiment --study uct-mcts --games-per-seat 10 --iterations 150 --output-dir results/uct_mcts
```

Try easy/medium/hard levels and compare their win rates and calculation times:

```bash
python run.py match --p1 easy --p2 hard
python run.py experiment --study difficulties --games-per-seat 8 --output-dir results/difficulties
```

Play as a human in the FAMNIT graphical interface (requires desktop GUI):

```bash
python run.py human --difficulty easy --human-seat 1
```

Mouse: choose green-highlighted source, destination, and capture position (as applicable). Esc restarts the move; close window to quit. For complete instructions and implementation details, read `README.md`.
