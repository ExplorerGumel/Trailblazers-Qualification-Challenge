"""
Prediction and inference functions.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


class Predictor:
    """Handle predictions on new data."""
    
    def __init__(self, model_path=None, model=None):
        """
        Initialize Predictor.
        
        Args:
            model_path (str, optional): Path to saved model
            model (Model, optional): Trained Keras model
        """
        if model_path:
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        elif model:
            self.model = model
        else:
            self.model = None
            print("No model provided!")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array): Features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model available for prediction!")
        
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def predict_with_confidence(self, X, n_iterations=100, dropout=True):
        """
        Make predictions with uncertainty estimation using MC Dropout.
        
        Args:
            X (array): Features
            n_iterations (int): Number of MC iterations
            dropout (bool): Whether to use dropout during prediction
            
        Returns:
            Mean predictions and standard deviations
        """
        if not dropout:
            return self.predict(X), np.zeros(len(X))
        
        predictions = []
        for _ in range(n_iterations):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def batch_predict(self, X, batch_size=32):
        """
        Make predictions in batches for large datasets.
        
        Args:
            X (array): Features
            batch_size (int): Batch size for prediction
            
        Returns:
            Array of predictions
        """
        predictions = self.model.predict(X, batch_size=batch_size)
        return predictions.flatten()
    
    def save_predictions(self, predictions, output_path, ids=None):
        """
        Save predictions to CSV file.
        
        Args:
            predictions (array): Predictions
            output_path (str): Path to save CSV
            ids (array, optional): Sample IDs
        """
        if ids is not None:
            df = pd.DataFrame({
                'id': ids,
                'prediction': predictions
            })
        else:
            df = pd.DataFrame({
                'prediction': predictions
            })
        
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
