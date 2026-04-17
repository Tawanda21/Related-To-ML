# Related-To-ML

This repository contains coursework and practical implementations for an Intelligent Systems / Machine Learning toolkit.

The goal is to build, compare, and test core AI approaches across different categories including:

- Rule-based systems
- Search algorithms
- Bayesian methods
- Decision trees
- Linear regression
- Neural networks

## Project Structure

```text
README.md
Intelligent Systems Toolkit/
	requirements.txt
	setup.py
	data/
		generate_sample_data.py
		raw/
		processed/
	docs/
	examples/
	notebooks/
	src/
		bayesian/
		decision trees/
		linear regression/
		neural networks/
		rule based/
		search algorithms/
		utils/
	tests/
```

### What Each Folder Is For

- `data/raw/`: Original datasets before cleaning.
- `data/processed/`: Cleaned or transformed datasets used for training/testing.
- `data/generate_sample_data.py`: Script for creating synthetic data for experiments.
- `docs/`: Project notes, method explanations, and write-ups.
- `examples/`: Small runnable demos of individual algorithms.
- `notebooks/`: Exploratory analysis, visualization, and model experiments.
- `src/`: Source code grouped by intelligent system technique.
- `src/utils/`: Shared helpers (metrics, preprocessing, plotting, etc.).
- `tests/`: Unit and integration tests for algorithms and utilities.

## Development Setup

### 1. Create and Activate a Virtual Environment

From the repository root:

```bash
python -m venv "Intelligent Systems Toolkit/.venv"
```

Activate it:

- Windows (PowerShell)

	```powershell
	.\"Intelligent Systems Toolkit"\.venv\Scripts\Activate.ps1
	```

- Windows (Command Prompt)

	```cmd
	"Intelligent Systems Toolkit\.venv\Scripts\activate.bat"
	```

- Linux/macOS

	```bash
	source "Intelligent Systems Toolkit/.venv/bin/activate"
	```

### 2. Install Requirements

```bash
pip install -r "Intelligent Systems Toolkit/requirements.txt"
```

## Planned Workflow

1. Start with shared utilities and data generation.
2. Implement baseline models in each `src/` category.
3. Add examples and notebooks for demonstration.
4. Add tests for correctness and regression protection.
5. Compare model behavior and document results in `docs/`.

## Notes

- `setup.py` is currently present as a package scaffold and can be expanded once module boundaries are finalized.
- Directory names under `src/` currently include spaces; this is fine for organization, but Python package imports are easier if folder names are later normalized (for example `decision_trees`, `linear_regression`, `rule_based`).
