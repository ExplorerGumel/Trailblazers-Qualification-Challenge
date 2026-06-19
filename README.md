# Trailblazers Qualification Challenge

This repository contains a regression model training pipeline and helper scripts for the Trailblazers qualification challenge.

## Setup

1. Install Miniconda (recommended) and create the environment:

```powershell
conda create -n trailblazers python=3.11 -y
conda activate trailblazers
pip install -r requirements.txt
```

2. (Optional) If you prefer conda packages for heavy ML libs:

```powershell
conda install -c conda-forge lightgbm
```

## Files

- `main.py` - Original neural network training pipeline using TensorFlow/Keras.
- `train_improved.py` - Experimental tree-based model pipeline with ensemble and tuning support.
- `run_quick.py` - Convenience wrapper for running the improved tree-based pipeline.
- `src/tree_pipeline.py` - Shared tree-based experiment module for HistGradientBoosting, RandomForest, and optional LightGBM.
- `scripts/` - CLI wrappers for convenience and reproducible execution.
- `src/` - Refactored internal modules (`data.py`, `features.py`, `models.py`, `utils.py`, `config.py`).
- `requirements.txt` - Python dependencies.
- `tests/` - Unit tests (run with `pytest`).

## Run

### Neural network workflow

This pipeline is designed for the original deep learning experiment:

```powershell
python main.py --train_path "C:\Users\Administrator\Downloads\Data\Train.csv" --test_path "C:\Users\Administrator\Downloads\Data\Test.csv"
```

### Tree-based experiment workflow

This pipeline showcases tree model experimentation with gradient boosting, random forest, and optional LightGBM support:

```powershell
python train_improved.py --train "C:\Users\Administrator\Downloads\Data\Train.csv"
```

If LightGBM is installed, it will be included in the candidate model set automatically.

### Package-style invocation

If you want to import the shared modules in `src/`:

```python
from src import run_tree_experiment

result = run_tree_experiment(
    train_path='data/raw/Train.csv',
    target_col='target',
    output_dir='models'
)
print(result)
```

### Wrapper scripts

Use the script wrappers for reproducible, command-line-friendly execution.

```powershell
python scripts/train_improved.py --train "C:\Users\Administrator\Downloads\Data\Train.csv"
python scripts/run_quick.py --train "C:\Users\Administrator\Downloads\Data\Train.csv"
```

`train_improved.py` is the main experiment entrypoint, while `run_quick.py` is a lightweight convenience wrapper.

## CI

A GitHub Actions workflow is included at `.github/workflows/python-app.yml`. It installs dependencies and runs tests via `pytest`.

## Notes

- Large data and model artifacts are ignored by `.gitignore`.
- If you want the CI to run the full training, modify the workflow to use smaller sample data or set resource/time limits.

If you'd like, I can make the CI run a lightweight smoke test instead of full `pytest`, or expand the README with examples and badges.