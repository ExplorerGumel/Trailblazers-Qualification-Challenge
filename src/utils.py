"""
Utility functions.
"""

import os
import numpy as np
import random
import tensorflow as tf
import joblib


def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_model(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load model from disk."""
    return tf.keras.models.load_model(path)


def save_scaler(scaler, path):
    """Save scaler to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path):
    """Load scaler from disk."""
    return joblib.load(path)
