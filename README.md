# Trailblazers Qualification Challenge

A reproducible tabular-regression project for the Trailblazers qualification challenge.

## What is included

- A tree-based training pipeline with train-only preprocessing, validation metrics, and persisted inference artifacts.
- An optional TensorFlow/Keras neural-network workflow in `main.py`.
- Unit tests and GitHub Actions CI for the lightweight tree workflow.

## Setup

Create an environment with Python 3.10 or 3.11, then install the training dependencies:

```powershell
pip install -r requirements.txt
```

For the fast CI/test dependency set only:

```powershell
pip install -r requirements-ci.txt
```

## Train a tree model

The supported training entry point is:

```powershell
python train_improved.py --train data/raw/Train.csv --target-col target --output-dir models
```

Add `--no-tune` to skip randomized hyperparameter search. The run reports R², MAE, and RMSE on a held-out validation split and saves one self-contained artifact:

```text
models/tree_regressor.joblib
```

The artifact includes the fitted model, feature order, missing-value statistics, and categorical encodings. Preprocessing is fit only on the training split, preventing validation leakage.

## Inference

Use the saved artifact with a CSV that contains the original feature columns (the target column is not required):

```python
from src.tree_pipeline import predict_tree_data

predictions = predict_tree_data(
    "models/tree_regressor.joblib",
    "data/raw/Test.csv",
)
```

Unseen categorical values are encoded safely and missing required feature columns produce a clear error.

## Test and lint

```powershell
ruff check src tests
pytest -q
```

The GitHub Actions workflow runs the same syntax, import, lint, and test checks on Python 3.10 and 3.11.

## Neural-network workflow

`main.py` remains available for the original TensorFlow/Keras experiment:

```powershell
python main.py --train_path data/raw/Train.csv --test_path data/raw/Test.csv
```

It requires TensorFlow, which is included in the full `requirements.txt` environment.
