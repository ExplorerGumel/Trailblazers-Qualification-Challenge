"""
Feature engineering functions.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class FeatureEngineer:
    """Handle feature engineering operations."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.kmeans_model = None
        
    def create_geo_clusters(self, df, lat_col='latitude', lon_col='longitude', 
                           n_clusters=5, random_state=42):
        """
        Create geographical clusters using KMeans on latitude/longitude.
        
        Args:
            df (DataFrame): Input DataFrame
            lat_col (str): Name of latitude column
            lon_col (str): Name of longitude column
            n_clusters (int): Number of clusters to create
            random_state (int): Random seed
            
        Returns:
            DataFrame with added cluster column
        """
        df = df.copy()
        
        if lat_col in df.columns and lon_col in df.columns:
            coords = df[[lat_col, lon_col]].values
            
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(
                    n_clusters=n_clusters, 
                    random_state=random_state
                )
                df['geo_cluster'] = self.kmeans_model.fit_predict(coords)
            else:
                df['geo_cluster'] = self.kmeans_model.predict(coords)
            
            print(f"Created {n_clusters} geographical clusters")
        else:
            print(f"Warning: {lat_col} or {lon_col} not found in DataFrame")
        
        return df
    
    def create_interaction_features(self, df, feature_pairs):
        """
        Create interaction features between pairs of columns.
        
        Args:
            df (DataFrame): Input DataFrame
            feature_pairs (list): List of tuples of column names
            
        Returns:
            DataFrame with added interaction features
        """
        df = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                print(f"Created interaction: {col1}_x_{col2}")
        
        return df
    
    def create_polynomial_features(self, df, columns, degree=2):
        """
        Create polynomial features for specified columns.
        
        Args:
            df (DataFrame): Input DataFrame
            columns (list): List of column names
            degree (int): Degree of polynomial
            
        Returns:
            DataFrame with added polynomial features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f'{col}_pow_{d}'] = df[col] ** d
                print(f"Created polynomial features for {col} up to degree {degree}")
        
        return df
    
    def create_aggregated_features(self, df, group_col, agg_cols, agg_funcs=['mean', 'std']):
        """
        Create aggregated features based on grouping.
        
        Args:
            df (DataFrame): Input DataFrame
            group_col (str): Column to group by
            agg_cols (list): Columns to aggregate
            agg_funcs (list): Aggregation functions
            
        Returns:
            DataFrame with added aggregated features
        """
        df = df.copy()
        
        if group_col in df.columns:
            for col in agg_cols:
                if col in df.columns:
                    for func in agg_funcs:
                        agg_df = df.groupby(group_col)[col].transform(func)
                        df[f'{col}_{func}_by_{group_col}'] = agg_df
                        print(f"Created: {col}_{func}_by_{group_col}")
        
        return df
    
    def create_distance_features(self, df, lat_col='latitude', lon_col='longitude', 
                                ref_lat=0, ref_lon=0):
        """
        Create distance features from a reference point.
        
        Args:
            df (DataFrame): Input DataFrame
            lat_col (str): Name of latitude column
            lon_col (str): Name of longitude column
            ref_lat (float): Reference latitude
            ref_lon (float): Reference longitude
            
        Returns:
            DataFrame with added distance feature
        """
        df = df.copy()
        
        if lat_col in df.columns and lon_col in df.columns:
            # Euclidean distance (simplified)
            df['distance_from_ref'] = np.sqrt(
                (df[lat_col] - ref_lat)**2 + (df[lon_col] - ref_lon)**2
            )
            print("Created distance feature from reference point")
        
        return df
    
    def select_features(self, df, exclude_cols=None):
        """
        Select features for modeling, excluding specified columns.
        
        Args:
            df (DataFrame): Input DataFrame
            exclude_cols (list): Columns to exclude
            
        Returns:
            DataFrame with selected features
        """
        if exclude_cols is None:
            exclude_cols = []
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Selected {len(feature_cols)} features")
        
        return df[feature_cols]
