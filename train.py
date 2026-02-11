"""
Model training and evaluation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
