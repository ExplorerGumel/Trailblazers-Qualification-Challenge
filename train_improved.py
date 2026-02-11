"""
Improved training script using scikit-learn ensembles.

Run:
    python train_improved.py --train "C:\\Users\\Administrator\\Downloads\\Data\\Train.csv" \
        --test "C:\\Users\\Administrator\\Downloads\\Data\\Test.csv"

This trains a HistGradientBoostingRegressor and RandomForestRegressor,
compares performance on a validation split, and saves the best model.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception:
    HAS_LGB = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures


def load_and_preprocess(train_path):
    df = pd.read_csv(train_path)
    # assume target named 'target' (if different, adjust)
    if 'target' not in df.columns:
        raise ValueError("Expected 'target' column in training data")

    # Simple preprocessing: fill numeric NAs with median, encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'target']

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.copy()

    encoders = {}
    for c in cat_cols:
        df[c] = df[c].fillna('NA')
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    X = df.drop(columns=['target'])
    y = df['target'].values
    return X, y, encoders


def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return {'r2': r2, 'mae': mae, 'rmse': rmse}


def main(args):
    X, y, encoders = load_and_preprocess(args.train)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    # Candidate models
    models = {
        'hgb': HistGradientBoostingRegressor(random_state=42, max_iter=500),
        'rf': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    }
    if HAS_LGB:
        models['lgb'] = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)

    results = {}
    best = {'name': None, 'score': -np.inf, 'model': None}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_val, y_val)
        results[name] = metrics
        if metrics['r2'] > best['score']:
            best = {'name': name, 'score': metrics['r2'], 'model': model}

    print('\nBest model:', best['name'], f"R2={best['score']:.4f}")

    # If baseline R2 < 0.95, attempt a quick randomized hyperparameter search for HGB
    if best['score'] < 0.95:
        print('\nRunning quick RandomizedSearchCV for HistGradientBoostingRegressor...')
        param_dist = {
            'learning_rate': stats.loguniform(1e-3, 1e-1),
            'max_iter': [200, 300, 500, 800],
            'max_depth': [3, 5, 8, None],
            'min_samples_leaf': [1, 3, 5, 10],
            'l2_regularization': stats.loguniform(1e-6, 1e-1)
        }
        hgb = HistGradientBoostingRegressor(random_state=42)
        search = RandomizedSearchCV(
            hgb, param_distributions=param_dist,
            n_iter=20, scoring='r2', cv=3, random_state=42, n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        print('Best params from RandomizedSearchCV:', search.best_params_)
        tuned = search.best_estimator_
        tuned_metrics = evaluate_model('hgb_tuned', tuned, X_val, y_val)
        # If tuned model improves, save it
        if tuned_metrics['r2'] > best['score']:
            best = {'name': 'hgb_tuned', 'score': tuned_metrics['r2'], 'model': tuned}

    # Try log-transforming the target and retraining HGB (sometimes helps with skewed targets)
    try:
        print('\nTrying log1p target transform with HGB...')
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        hgb_log = HistGradientBoostingRegressor(random_state=42, **(search.best_params_ if 'search' in locals() else {}))
        hgb_log.fit(X_train, y_train_log)
        y_pred_log = hgb_log.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        r2_log = r2_score(y_val, y_pred)
        mae_log = mean_absolute_error(y_val, y_pred)
        rmse_log = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"hgb_log -> R2: {r2_log:.4f}, MAE: {mae_log:.4f}, RMSE: {rmse_log:.4f}")
        if r2_log > best['score']:
            best = {'name': 'hgb_log', 'score': r2_log, 'model': hgb_log}
    except Exception as e:
        print('Log-target training failed:', e)

    # Try simple averaging ensemble between tuned HGB and RF
    try:
        print('\nTrying simple average ensemble of tuned HGB and RF...')
        models_for_ens = []
        if 'tuned' in locals():
            models_for_ens.append(tuned)
        models_for_ens.append(models['rf'])
        ensemble_preds = np.zeros(len(X_val))
        for m in models_for_ens:
            ensemble_preds += m.predict(X_val)
        ensemble_preds /= len(models_for_ens)
        r2_ens = r2_score(y_val, ensemble_preds)
        mae_ens = mean_absolute_error(y_val, ensemble_preds)
        rmse_ens = np.sqrt(mean_squared_error(y_val, ensemble_preds))
        print(f"ensemble_avg -> R2: {r2_ens:.4f}, MAE: {mae_ens:.4f}, RMSE: {rmse_ens:.4f}")
        if r2_ens > best['score']:
            best = {'name': 'ensemble_avg', 'score': r2_ens, 'model': None}
    except Exception as e:
        print('Ensembling failed:', e)

    # Try polynomial feature expansion (degree=2) + SelectKBest, then retrain HGB
    try:
        print('\nTrying polynomial features + feature selection...')
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_train_num = X_train[numeric_cols]
        X_val_num = X_val[numeric_cols]
        X_train_poly = poly.fit_transform(X_train_num)
        X_val_poly = poly.transform(X_val_num)

        # combine with categorical columns (if any)
        cat_cols = [c for c in X_train.columns if c not in numeric_cols]
        if cat_cols:
            X_train_cat = X_train[cat_cols].to_numpy()
            X_val_cat = X_val[cat_cols].to_numpy()
            X_train_exp = np.hstack([X_train_poly, X_train_cat])
            X_val_exp = np.hstack([X_val_poly, X_val_cat])
        else:
            X_train_exp = X_train_poly
            X_val_exp = X_val_poly

        selector = SelectKBest(score_func=f_regression, k=min(300, X_train_exp.shape[1]))
        X_train_sel = selector.fit_transform(X_train_exp, y_train)
        X_val_sel = selector.transform(X_val_exp)

        hgb_poly = HistGradientBoostingRegressor(random_state=42, **(search.best_params_ if 'search' in locals() else {}))
        hgb_poly.fit(X_train_sel, y_train)
        y_pred = hgb_poly.predict(X_val_sel)
        r2_poly = r2_score(y_val, y_pred)
        mae_poly = mean_absolute_error(y_val, y_pred)
        rmse_poly = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"hgb_poly -> R2: {r2_poly:.4f}, MAE: {mae_poly:.4f}, RMSE: {rmse_poly:.4f}")
        if r2_poly > best['score']:
            best = {'name': 'hgb_poly', 'score': r2_poly, 'model': hgb_poly}
    except Exception as e:
        print('Polynomial features step failed:', e)

    # Save best model and preprocessing objects
    os.makedirs('models', exist_ok=True)
    joblib.dump(best['model'], os.path.join('models', f"best_model_{best['name']}.joblib"))
    joblib.dump({'scaler': scaler, 'encoders': encoders, 'numeric_cols': numeric_cols}, os.path.join('models', 'preprocessing.joblib'))
