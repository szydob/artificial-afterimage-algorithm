# artificial-afterimage-algorithm
Group project for the Deep Learning and Computational Intelligence course (2026)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or if you have pip already installed:

```bash
pip install uv
```

### Create Environment

```bash
uv sync
```

### Add Jupyter Kernel so you can use it in notebook

```bash
uv run python -m ipykernel install --user --name=aaia --display-name "AAIA (venv)"
```

## Run app

```bash
uv run python main.py
```

Alternative:

```bash
uv run streamlit run gui.py
```

## What the app does

- Loads a CSV and applies sidebar filters.
- Runs AAIA to find one representative solution (center-like point).
- Computes Manhattan distance of each sample to this solution.
- Returns nearest samples by selected percentage or count.
- Lets you tune AAIA `population_size` and `max_iterations`.
- Supports early stopping (`early_stopping_rounds`) and improvement tolerance (`tol`).
- Shows histogram + scatter side-by-side (scatter uses 2 user-selected features).
- Download the top-N nearest samples as CSV directly from the UI.

## Typical flow

1. Upload a CSV.
2. Pick features and filter ranges in the sidebar.
3. Set AAIA parameters (`iterations`, `population`, optional early stopping).
4. Run clustering.
5. Explore nearest samples, centroid history, histogram, and scatter.

## Project structure

```
.
├── aaia.py           # AAIA algorithm implementation
├── gui.py            # Streamlit UI
├── main.py           # app entrypoint (starts Streamlit)
├── pyproject.toml    # project dependencies
└── README.md
```

## References

Implementation of the **Artificial Afterimage Algorithm (AAIA)** — a bio-inspired metaheuristic for data clustering, based on:

> Demir, Murat. 2025. "Artificial Afterimage Algorithm: A New Bio-Inspired Metaheuristic Algorithm and Its Clustering Application" Applied Sciences 15, no. 3: 1359. https://doi.org/10.3390/app15031359