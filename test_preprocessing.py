"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.data import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': ['A', 'B', 'A', 'C', 'B'],
            'target': [100, 200, 300, 400, 500]
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance."""
        return DataPreprocessor()
    
    def test_handle_missing_values_mean(self, preprocessor, sample_data):
        """Test missing value handling with mean strategy."""
        result = preprocessor.handle_missing_values(sample_data, strategy='mean')
        assert result['feature1'].isnull().sum() == 0
        assert result['feature1'].iloc[2] == 3.0  # Mean of [1, 2, 4, 5]
    
    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding."""
        result = preprocessor.encode_categorical(sample_data, ['feature3'])
        assert result['feature3'].dtype in [np.int32, np.int64]
        assert len(result['feature3'].unique()) == 3  # A, B, C
    
    def test_scale_features(self, preprocessor):
        """Test feature scaling."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[2, 3], [4, 5]])
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(
            X_train, X_test, method='standard'
        )
        
        # Check that scaling was applied
        assert X_train_scaled.mean(axis=0).sum() < 0.001  # Close to 0
        assert abs(X_train_scaled.std(axis=0).sum() - 2) < 0.001  # Close to 1 per feature
    
    def test_split_data(self, preprocessor):
        """Test data splitting."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        X_train, X_val, y_train, y_val = preprocessor.split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        assert X_train.shape[0] == 80
        assert X_val.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_val.shape[0] == 20


if __name__ == '__main__':
    pytest.main([__file__])
