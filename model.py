"""
TensorFlow model building and architecture definitions.
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
