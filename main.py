"""
Main training script for the regression model.

Usage:
    python main.py --train_path data/raw/Train.csv --test_path data/raw/Test.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import seed_everything, save_model
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import RegressionModel, ModelTrainer
from src import config


def main(args):
    """Main training pipeline."""
    
    # Set random seed for reproducibility
    seed_everything(config.RANDOM_SEED)
    
    print("="*50)
    print("REGRESSION MODEL TRAINING PIPELINE")
    print("="*50)
    
    # 1. Load Data
    print("\n[1/6] Loading data...")
    loader = DataLoader(args.train_path, args.test_path)
    train_data, test_data = loader.load_data()
    
    # 2. Preprocess Data
    print("\n[2/6] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Combine train and test for unified preprocessing
    combined_data, train_len = loader.get_combined_data(target_col='target')
    
    # Handle missing values
    combined_data = preprocessor.handle_missing_values(combined_data, strategy='mean')
    
    # Identify categorical columns (modify based on your data)
    categorical_cols = combined_data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        combined_data = preprocessor.encode_categorical(combined_data, categorical_cols)
    
    # 3. Feature Engineering
    print("\n[3/6] Engineering features...")
    engineer = FeatureEngineer()
    
    # Create geographical clusters if lat/lon exist
    if 'latitude' in combined_data.columns and 'longitude' in combined_data.columns:
        combined_data = engineer.create_geo_clusters(
            combined_data, 
            n_clusters=config.N_CLUSTERS
        )
    
    # Split back into train and test
    train_data = combined_data[:train_len].copy()
    test_data = combined_data[train_len:].copy()
    
    # Separate features and target
    exclude_cols = ['target']  # Add other columns to exclude
    X = engineer.select_features(train_data, exclude_cols=exclude_cols)
    y = train_data['target'].values
    
    # 4. Split and Scale
    print("\n[4/6] Splitting and scaling data...")
    X_train, X_val, y_train, y_val = preprocessor.split_data(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED
    )
    
    X_train_scaled, X_val_scaled = preprocessor.scale_features(
        X_train, X_val, method='standard'
    )
    
    # 5. Build and Train Model
    print("\n[5/6] Building and training model...")
    input_dim = X_train_scaled.shape[1]
    
    model = RegressionModel(input_dim)
    model.build_simple_nn(
        layers=[
            config.NN_ARCHITECTURE['layer_1'],
            config.NN_ARCHITECTURE['layer_2'],
            config.NN_ARCHITECTURE['layer_3']
        ],
        dropout_rate=config.NN_ARCHITECTURE['dropout_rate'],
        activation=config.NN_ARCHITECTURE['activation']
    )
    
    model.compile_model(
        learning_rate=config.MODEL_CONFIG['learning_rate'],
        loss='mse',
        metrics=config.METRICS
    )
    
    model.get_summary()
    
    # Get callbacks
    callbacks = model.get_callbacks(
        checkpoint_path=os.path.join(config.MODELS_DIR, 'best_model.keras'),
        early_stopping=config.USE_EARLY_STOPPING,
        reduce_lr=config.USE_REDUCE_LR,
        model_checkpoint=config.USE_MODEL_CHECKPOINT
    )
    
    # Train model
    trainer = ModelTrainer(model)
    history = trainer.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=config.MODEL_CONFIG['epochs'],
        batch_size=config.MODEL_CONFIG['batch_size'],
        callbacks=callbacks
    )
    
    # 6. Evaluate Model
    print("\n[6/6] Evaluating model...")
    metrics = trainer.evaluate(X_val_scaled, y_val)
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(config.REPORTS_DIR, 'figures', 'training_history.png')
    )
    
    # Plot predictions
    y_pred = trainer.predict(X_val_scaled)
    trainer.plot_predictions(
        y_val, y_pred,
        save_path=os.path.join(config.REPORTS_DIR, 'figures', 'predictions.png')
    )
    
    # Save final model
    save_model(model.model, os.path.join(config.MODELS_DIR, 'final_model'))
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train regression model')
    parser.add_argument('--train_path', type=str, 
                       default='data/raw/Train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--test_path', type=str,
                       default='data/raw/Test.csv',
                       help='Path to test data CSV')
    
    args = parser.parse_args()
    
    model, metrics = main(args)
