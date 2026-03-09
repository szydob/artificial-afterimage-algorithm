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

## Project structure

```
.
├── aaia.py           # AAIA algorithm implementation
├── pyproject.toml    # project dependencies
└── README.md
```

## References

Implementation of the **Artificial Afterimage Algorithm (AAIA)** — a bio-inspired metaheuristic for data clustering, based on:

> Demir, Murat. 2025. "Artificial Afterimage Algorithm: A New Bio-Inspired Metaheuristic Algorithm and Its Clustering Application" Applied Sciences 15, no. 3: 1359. https://doi.org/10.3390/app15031359