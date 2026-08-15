"""Reliable tree-based regression training and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import scipy.stats as stats

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


@dataclass
class TreePreprocessor:
    """Fit deterministic tabular preprocessing on training features only."""

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_fill_values: dict[str, float] = field(default_factory=dict)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, features: pd.DataFrame) -> "TreePreprocessor":
        self.numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = features.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        supported = set(self.numeric_columns + self.categorical_columns)
        unsupported = [column for column in features.columns if column not in supported]
        if unsupported:
            raise ValueError(
                "Unsupported feature dtypes for columns: " + ", ".join(unsupported)
            )

        self.numeric_fill_values = {
            column: float(features[column].median())
            if not pd.isna(features[column].median())
            else 0.0
            for column in self.numeric_columns
        }
        self.category_maps = {}
        for column in self.categorical_columns:
            values = features[column].fillna("__MISSING__").astype(str)
            self.category_maps[column] = {
                value: index for index, value in enumerate(sorted(values.unique()))
            }
        return self

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_names if column not in features.columns]
        if missing:
            raise ValueError("Missing required feature columns: " + ", ".join(missing))

        transformed = pd.DataFrame(index=features.index)
        for column in self.numeric_columns:
            values = pd.to_numeric(features[column], errors="coerce")
            transformed[column] = values.fillna(self.numeric_fill_values[column])
        for column in self.categorical_columns:
            values = features[column].fillna("__MISSING__").astype(str)
            transformed[column] = values.map(self.category_maps[column]).fillna(-1).astype(int)
        return transformed[self.feature_names]


def load_tree_data(train_path: str, target_col: str = "target") -> pd.DataFrame:
    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in training data")
    if df.empty:
        raise ValueError("Training data is empty")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "target"
) -> tuple[pd.DataFrame, np.ndarray]:
    target = pd.to_numeric(df[target_col], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Target column '{target_col}' must contain only numeric, non-null values")
    features = df.drop(columns=[target_col])
    if features.empty:
        raise ValueError("Training data must contain at least one feature column")
    return features, target.to_numpy()


def preprocess_tree_data(
    df: pd.DataFrame, target_col: str = "target"
) -> tuple[pd.DataFrame, np.ndarray, TreePreprocessor, list[str]]:
    """Compatibility helper that fits and transforms a single training frame."""
    features, target = split_features_target(df, target_col)
    preprocessor = TreePreprocessor().fit(features)
    return preprocessor.transform(features), target, preprocessor, preprocessor.numeric_columns


def evaluate_model(
    name: str, model: Any, features: pd.DataFrame, target: np.ndarray
) -> dict[str, float]:
    predictions = model.predict(features)
    results = {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
    }
    print(
        f"{name} -> R2: {results['r2']:.4f}, "
        f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}"
    )
    return results


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "hgb": HistGradientBoostingRegressor(random_state=random_state, max_iter=500),
        "rf": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
    }
    if HAS_LGBM:
        models["lgb"] = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def run_tree_experiment(
    train_path: str,
    target_col: str = "target",
    output_dir: str = "models",
    random_state: int = 42,
    tune: bool = True,
) -> dict[str, Any]:
    """Train candidate regressors and persist the winning model with preprocessing."""
    df = load_tree_data(train_path, target_col=target_col)
    raw_features, target = split_features_target(df, target_col=target_col)
    if len(raw_features) < 5:
        raise ValueError("At least five rows are required for a train/validation split")

    raw_train, raw_validation, y_train, y_validation = train_test_split(
        raw_features, target, test_size=0.2, random_state=random_state
    )
    preprocessor = TreePreprocessor().fit(raw_train)
    X_train = preprocessor.transform(raw_train)
    X_validation = preprocessor.transform(raw_validation)

    models = build_candidate_models(random_state=random_state)
    results: dict[str, dict[str, float]] = {}
    best_name: str | None = None
    best_model: Any = None
    best_score = -np.inf

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_validation, y_validation)
        results[name] = metrics
        if metrics["r2"] > best_score:
            best_name, best_model, best_score = name, model, metrics["r2"]

    if tune and len(X_train) >= 15 and best_score < 0.95:
        print("Running randomized search for HistGradientBoostingRegressor...")
        search = RandomizedSearchCV(
            HistGradientBoostingRegressor(random_state=random_state),
            param_distributions={
                "learning_rate": stats.loguniform(1e-3, 1e-1),
                "max_iter": [200, 300, 500, 800],
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [1, 3, 5, 10],
                "l2_regularization": stats.loguniform(1e-6, 1e-1),
            },
            n_iter=20,
            scoring="r2",
            cv=min(3, len(X_train)),
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_metrics = evaluate_model(
            "hgb_tuned", search.best_estimator_, X_validation, y_validation
        )
        results["hgb_tuned"] = tuned_metrics
        if tuned_metrics["r2"] > best_score:
            best_name, best_model, best_score = (
                "hgb_tuned",
                search.best_estimator_,
                tuned_metrics["r2"],
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "tree_regressor.joblib"
    joblib.dump(
        {
            "model": best_model,
            "preprocessor": preprocessor,
            "target_col": target_col,
            "feature_names": preprocessor.feature_names,
        },
        artifact_path,
    )
    print(f"Best model artifact saved to {artifact_path}")

    return {
        "best_name": best_name,
        "best_score": float(best_score),
        "results": results,
        "artifact_path": str(artifact_path),
    }


def predict_tree_data(model_path: str, data_path: str) -> np.ndarray:
    """Load a persisted tree artifact and predict from a CSV file."""
    artifact = joblib.load(model_path)
    required = {"model", "preprocessor", "feature_names"}
    if not required.issubset(artifact):
        raise ValueError("Invalid tree model artifact")
    features = pd.read_csv(data_path)
    transformed = artifact["preprocessor"].transform(features)
    return artifact["model"].predict(transformed)
