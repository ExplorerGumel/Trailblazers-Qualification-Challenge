"""
Models and training module.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionModel:
    """Build and configure regression models."""
    
    def __init__(self, input_dim):
        """
        Initialize RegressionModel.
        
        Args:
            input_dim (int): Number of input features
        """
        self.input_dim = input_dim
        self.model = None
        self.history = None
        
    def build_simple_nn(self, layers=[128, 64, 32], dropout_rate=0.3, 
                       activation='relu'):
        """
        Build a simple feedforward neural network.
        
        Args:
            layers (list): List of neurons in each hidden layer
            dropout_rate (float): Dropout rate
            activation (str): Activation function
            
        Returns:
            Compiled Keras model
        """
        self.model = Sequential([Input(shape=(self.input_dim,))])
        
        for units in layers:
            self.model.add(Dense(units, activation=activation))
            self.model.add(Dropout(dropout_rate))
        
        # Output layer for regression
        self.model.add(Dense(1, activation='linear'))
        
        print(f"Built simple NN with architecture: {layers}")
        return self.model
    
    def build_deep_nn(self, layers=[256, 128, 64, 32], dropout_rate=0.3,
                     use_batch_norm=True, activation='relu'):
        """
        Build a deeper neural network with batch normalization.
        
        Args:
            layers (list): List of neurons in each hidden layer
            dropout_rate (float): Dropout rate
            use_batch_norm (bool): Whether to use batch normalization
            activation (str): Activation function
            
        Returns:
            Compiled Keras model
        """
        self.model = Sequential([Input(shape=(self.input_dim,))])
        
        for units in layers:
            self.model.add(Dense(units, activation=activation))
            
            if use_batch_norm:
                self.model.add(BatchNormalization())
            
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, activation='linear'))
        
        print(f"Built deep NN with architecture: {layers}")
        return self.model
    
    def compile_model(self, learning_rate=0.001, loss='mse', 
                     metrics=['mae', 'mse']):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate for optimizer
            loss (str): Loss function
            metrics (list): List of metrics to track
        """
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        print(f"Model compiled with lr={learning_rate}, loss={loss}")
    
    def get_callbacks(self, checkpoint_path='best_model.keras',
                     early_stopping=True, reduce_lr=True, 
                     model_checkpoint=True):
        """
        Get list of training callbacks.
        
        Args:
            checkpoint_path (str): Path to save best model
            early_stopping (bool): Whether to use early stopping
            reduce_lr (bool): Whether to use learning rate reduction
            model_checkpoint (bool): Whether to save best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        if reduce_lr:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            )
        
        if model_checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        print(f"Configured {len(callbacks)} callbacks")
        return callbacks
    
    def custom_scheduler(self, epoch, lr):
        """
        Custom learning rate scheduler.
        
        Args:
            epoch (int): Current epoch
            lr (float): Current learning rate
            
        Returns:
            Updated learning rate
        """
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    def get_summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet!")


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, model):
        """
        Initialize ModelTrainer.
        
        Args:
            model: RegressionModel instance
        """
        self.model = model
        self.history = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None,
             epochs=100, batch_size=32, callbacks=None, verbose=1):
        """
        Train the model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            X_val (array, optional): Validation features
            y_val (array, optional): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            callbacks (list): List of Keras callbacks
            verbose (int): Verbosity mode
            
        Returns:
            Training history
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test (array): Test features
            y_test (array): Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.model.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\n=== Evaluation Metrics ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (array): Features
            
        Returns:
            Predictions
        """
        predictions = self.model.model.predict(X)
        return predictions
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Train MAE')
            if 'val_mae' in self.history.history:
                axes[1].plot(self.history.history['val_mae'], label='Val MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Training and Validation MAE')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        """
        Plot true vs predicted values.
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].set_title('True vs Predicted Values')
        axes[0].grid(True)
        
        # Residuals
        residuals = y_true - y_pred.flatten()
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
