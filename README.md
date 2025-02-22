# Conway’s Game of Life Entropy Analysis

## Overview

This repository contains `golLab.py`, a Python application built with PyQt6 to analyze the complexity evolution in Conway’s Game of Life, a cellular automaton. The program visualizes and quantifies how simple, deterministic rules transform a high-entropy initial state (50% alive cells on a 30x30 grid) into stable, low-complexity attractors, exploring the hypothesis that entropy drives the creation of order in a deterministic universe. This work supports my thesis, *Entropy is the Engine of Creation: How Simple Rules Yield Complexity in a Deterministic Universe*, by investigating two key questions:

1. Does deterministic evolution preferentially select attractors with lower algorithmic complexity from high-entropy systems?
2. Are these attractors robust under perturbations and dominant in long-term behavior, even amidst chaotic dynamics?

The application uses the Hamming complexity (\(C_{HT}\)) metric, combining Hamming distance and transition counts, to measure disorder and track its simplification over 1,000 steps, with optional perturbations (flipping 5 random cells). It includes interactive plots, animations, and data aggregation for 10 runs to confirm stability and dominance.

## Installation

To set up and run this project, follow these steps:

### Clone the Repository:
```bash
git clone https://github.com/peterkelly70/golEntropy.git
cd golEntropy
```

### Set Up the Virtual Environment:
Use the provided `setup_env.sh` script to create a virtual environment (`gol_env`) and install dependencies:
```bash
./setup_env.sh
```
This script requires `virtualenv` (installable via `sudo apt install python3-virtualenv` on Linux) and uses `requirements.txt` to install `numpy`, `matplotlib`, `seaborn`, `PyQt6`, `pandas`, and `scipy`.

### Activate the Virtual Environment:
```bash
source gol_env/bin/activate
```

### Run the Application:
```bash
python golLab.py
```

### Deactivate the Environment (when done):
```bash
deactivate
```

## Usage

Launch `golLab.py` to open a GUI with four tabs:
- **Original C_HT Plot**
- **Perturbed C_HT Plot**
- **Grid Visualization**
- **Analysis Results**

### Controls:
- Adjust parameters (grid width, height, steps, alive probability) in the control panel.
- Use buttons like:
  - `Run Simulation`
  - `Collect Data (10 Runs)`
  - `Plot Aggregate Data`
  - `Toggle Visualization`
  - `Save Run`
  - `Load Run`
  - `Export Data as CSV`
  - `Apply Perturbation`
- Copy plots or analysis text to the clipboard, or save them as PNG files using the `Copy to Clipboard` and `Save as PNG` buttons in each tab.

## Concepts and Discussion

### Conway’s Game of Life

Conway’s Game of Life is a cellular automaton where each cell in a grid follows four simple rules based on its neighbors' states (alive or dead). Starting with a random 30x30 grid (50% alive, 50% dead), the game evolves deterministically, creating patterns like still lifes, oscillators, and gliders. This project investigates how these rules transform initial chaos into order, mirroring cosmic processes.

### Entropy and Complexity

- **Entropy:** In information theory, entropy measures disorder or uncertainty. For a binary string (e.g., the grid’s 900 cells), maximal entropy occurs when cells are equally likely to be alive (1) or dead (0)—50% probability in this case—maximizing randomness and unpredictability. As the game evolves, entropy decreases, reflecting the emergence of ordered, low-disorder patterns.

- **Hamming Complexity (C_{HT}):** This metric quantifies algorithmic complexity (disorder) by combining Hamming distance (difference from all 0s or 1s) and transition counts (changes between consecutive bits). The formula is:
  
  \[
  C_{HT} = ␇lpha \cdot H + (1 - ␇lpha) \cdot T
  \]
  
  where \(H\) is the normalized Hamming distance, \(T\) is the normalized transition count, and \(␇lpha = 0.5\). \(C_{HT}\) ranges from 0 (perfect order, e.g., all 0s or all 1s) to 0.5 (maximum disorder, e.g., alternating 0s and 1s). In `golLab.py`, \(C_{HT}\) tracks how complexity simplifies from ~0.5 (high entropy) to ~0 (low complexity, sparse patterns) over 1,000 steps.

### Connection to Dissipative Structures
The findings in `golLab.py` echo Ilya Prigogine’s work on dissipative structures, where complex, ordered patterns emerge from high-entropy systems far from equilibrium through energy dissipation. In Conway’s Game of Life, the deterministic rules serve as a computational analogue, transforming a maximally random, high-entropy grid into stable, low-complexity attractors via the simplification of complexity. This parallels Prigogine’s observation that entropy can drive order in systems like chemical reactions or biological processes, suggesting that the emergent order in Conway’s Game of Life reflects universal principles of self-organization seen in nature, from chemical oscillations to cosmic structures.

## Hypotheses and Findings

### Low-Complexity Attractors
The simulations show that deterministic rules preferentially select stable, sparse patterns (e.g., 33 live cells, 3.67% alive) with low \(C_{HT}\) (~0.053–0.074), supporting the hypothesis that high-entropy systems evolve into low-complexity attractors.

### Robustness and Dominance
Perturbations (flipping 5 cells) don’t disrupt this stabilization, and aggregates over 10 runs confirm dominance, with means stabilizing near 0 after ~800 steps and narrow standard deviation bands, indicating resilience and prevalence.

These findings align with my thesis, suggesting entropy drives creation by simplifying complexity through simple, deterministic rules, mirroring the universe’s transition from chaos to order.

## Files
- `golLab.py`: The main Python script with PyQt6 GUI for running and analyzing Conway’s Game of Life.
- `.gitignore`: Excludes `gol_env/`, `__pycache__/`, `*.pyc`, and `*.log` from version control.
- `requirements.txt`: Lists Python dependencies for installation.
- `setup_env.sh`: Bash script to set up the virtual environment and install dependencies.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests for improvements. For questions or collaboration, contact me via GitHub or my Substack article: *From Chaos to Order: How Simple Rules in Conway’s Game of Life Reveal Entropy’s Creative Power*.

## License

This project is licensed under the Creative Commons CC3 License - see the LICENSE file for details.

## Acknowledgments

Thanks to the xAI community and Conway’s Game of Life enthusiasts for inspiration. Special thanks to tools like PyQt6, matplotlib, and numpy for enabling this analysis.
Many Thanks to Grok,ChatGPT and Clude, whos help in formualting the text and writing a bunch of Python Code I had little interets in doing.
All ideas are mine, even if a bulk of the words aren't
