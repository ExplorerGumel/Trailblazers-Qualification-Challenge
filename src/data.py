"""
Data loading and preprocessing module.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class DataLoader:
    """Handle data loading operations."""
    
    def __init__(self, train_path, test_path=None):
        """
        Initialize DataLoader.
        
        Args:
            train_path (str): Path to training data CSV
            test_path (str, optional): Path to test data CSV
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """Load training and test datasets."""
        self.train_data = pd.read_csv(self.train_path)
        print(f"Train data loaded: {self.train_data.shape}")
        
        if self.test_path:
            self.test_data = pd.read_csv(self.test_path)
            print(f"Test data loaded: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def get_combined_data(self, target_col='target'):
        """
        Combine train and test data for unified preprocessing.
        
        Args:
            target_col (str): Name of target column
            
        Returns:
            Combined DataFrame and split index
        """
        train_len = len(self.train_data)
        
        if self.test_data is not None:
            # Add target column to test if it doesn't exist
            if target_col not in self.test_data.columns:
                self.test_data[target_col] = np.nan
            
            combined = pd.concat([self.train_data, self.test_data], 
                                axis=0, ignore_index=True)
        else:
            combined = self.train_data.copy()
        
        print(f"Combined data shape: {combined.shape}")
        return combined, train_len


class DataPreprocessor:
    """Handle all data preprocessing operations."""
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in DataFrame.
        
        Args:
            df (DataFrame): Input DataFrame
            strategy (str): Strategy for handling missing values
            
        Returns:
            DataFrame with handled missing values
        """
        if strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif strategy == 'drop':
            df = df.dropna()
            
        print(f"Missing values handled using {strategy} strategy")
        return df
    
    def encode_categorical(self, df, categorical_columns):
        """
        Encode categorical variables.
        
        Args:
            df (DataFrame): Input DataFrame
            categorical_columns (list): List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    df[col] = self.label_encoders[col].transform(
                        df[col].astype(str)
                    )
        
        print(f"Encoded {len(categorical_columns)} categorical columns")
        return df
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """
        Scale numerical features.
        
        Args:
            X_train (array): Training features
            X_test (array, optional): Test features
            method (str): Scaling method ('standard' or 'minmax')
            
        Returns:
            Scaled train and test features
        """
        if method == 'standard':
            scaler = self.standard_scaler
        elif method == 'minmax':
            scaler = self.minmax_scaler
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            print(f"Features scaled using {method} scaling")
            return X_train_scaled, X_test_scaled
        
        print(f"Features scaled using {method} scaling")
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and validation sets.
        
        Args:
            X (array): Features
            y (array): Target
            test_size (float): Proportion of validation data
            random_state (int): Random seed
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split: Train {X_train.shape}, Val {X_val.shape}")
        return X_train, X_val, y_train, y_val
