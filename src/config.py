"""
Configuration settings for the project.
"""

import os

# Random seed for reproducibility
RANDOM_SEED = 42

# Data paths
DATA_DIR = 'data/raw'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, 'figures'), exist_ok=True)

# Data preprocessing
TEST_SIZE = 0.2
N_CLUSTERS = 5

# Neural network architecture
NN_ARCHITECTURE = {
    'layer_1': 128,
    'layer_2': 64,
    'layer_3': 32,
    'dropout_rate': 0.3,
    'activation': 'relu'
}

# Model configuration
MODEL_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
}

# Training settings
USE_EARLY_STOPPING = True
USE_REDUCE_LR = True
USE_MODEL_CHECKPOINT = True

# Metrics
METRICS = ['mae', 'mse']
