import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats as stats

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def load_tree_data(train_path: str, target_col: str = 'target'):
    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in training data")
    return df


def preprocess_tree_data(df: pd.DataFrame, target_col: str = 'target'):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    encoders = {}
    for c in cat_cols:
        df[c] = df[c].fillna('NA').astype(str)
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y, encoders, num_cols


def scale_features(X_train, X_val, numeric_cols):
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val, scaler


def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    results = {
        'r2': r2_score(y_val, y_pred),
        'mae': mean_absolute_error(y_val, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred))
    }
    print(f"{name} -> R2: {results['r2']:.4f}, MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
    return results


def build_candidate_models(random_state: int = 42):
    models = {
        'hgb': HistGradientBoostingRegressor(random_state=random_state, max_iter=500),
        'rf': RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    }
    if HAS_LGBM:
        models['lgb'] = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=random_state, n_jobs=-1)
    return models


def run_tree_experiment(train_path: str,
                        test_path: str = None,
                        target_col: str = 'target',
                        output_dir: str = 'models',
                        random_state: int = 42,
                        tune: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    df = load_tree_data(train_path, target_col=target_col)
    X, y, encoders, numeric_cols = preprocess_tree_data(df, target_col=target_col)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    X_train, X_val, scaler = scale_features(X_train.copy(), X_val.copy(), numeric_cols)

    models = build_candidate_models(random_state=random_state)
    best_score = -np.inf
    best_name = None
    best_model = None
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_val, y_val)
        results[name] = metrics
        if metrics['r2'] > best_score:
            best_score = metrics['r2']
            best_name = name
            best_model = model

    if tune and best_score < 0.95:
        print('\nRunning randomized search for HistGradientBoostingRegressor...')
        param_dist = {
            'learning_rate': stats.loguniform(1e-3, 1e-1),
            'max_iter': [200, 300, 500, 800],
            'max_depth': [3, 5, 8, None],
            'min_samples_leaf': [1, 3, 5, 10],
            'l2_regularization': stats.loguniform(1e-6, 1e-1)
        }
        search = RandomizedSearchCV(
            HistGradientBoostingRegressor(random_state=random_state),
            param_distributions=param_dist,
            n_iter=20,
            scoring='r2',
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        tuned_metrics = evaluate_model('hgb_tuned', tuned, X_val, y_val)
        results['hgb_tuned'] = tuned_metrics
        if tuned_metrics['r2'] > best_score:
            best_score = tuned_metrics['r2']
            best_name = 'hgb_tuned'
            best_model = tuned

    model_path = os.path.join(output_dir, f'best_model_{best_name}.joblib')
    scaler_path = os.path.join(output_dir, 'preprocessing.joblib')
    joblib.dump(best_model, model_path)
    joblib.dump({'scaler': scaler, 'encoders': encoders, 'numeric_cols': numeric_cols}, scaler_path)

    print(f"Best model saved to {model_path}")
    print(f"Preprocessing objects saved to {scaler_path}")

    return {
        'best_name': best_name,
        'best_score': best_score,
        'results': results,
        'model_path': model_path,
        'scaler_path': scaler_path
    }
