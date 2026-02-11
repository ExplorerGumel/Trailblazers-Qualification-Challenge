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

- `main.py` - Original project training pipeline.
- `train_improved.py` - Improved ensemble training and experiments (scikit-learn + LightGBM).
- `run_quick.py` - Quick runner that invokes the improved pipeline.
- `scripts/` - CLI wrappers for convenience.
- `src/` - Refactored internal modules (`data.py`, `features.py`, `models.py`, `utils.py`, `config.py`).
- `requirements.txt` - Python dependencies.
- `tests/` - Unit tests (run with `pytest`).

## Run

Train with the improved script (uses local data):

```powershell
python train_improved.py --train "C:\Users\Administrator\Downloads\Data\Train.csv"
```

Run the original pipeline:

```powershell
python main.py --train_path "C:\Users\Administrator\Downloads\Data\Train.csv" --test_path "C:\Users\Administrator\Downloads\Data\Test.csv"
```

## CI

A GitHub Actions workflow is included at `.github/workflows/python-app.yml`. It installs dependencies and runs tests via `pytest`.

## Notes

- Large data and model artifacts are ignored by `.gitignore`.
- If you want the CI to run the full training, modify the workflow to use smaller sample data or set resource/time limits.

If you'd like, I can make the CI run a lightweight smoke test instead of full `pytest`, or expand the README with examples and badges.