"""
Advanced data preprocessing utilities for heart disease prediction.
Handles data cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from loguru import logger


class HeartDiseaseDataPreprocessor:
    """Advanced preprocessing pipeline for heart disease data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.feature_column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        self.target_column_name = 'class'
        
        self.preprocessing_pipeline: Optional[ColumnTransformer] = None
        self.feature_names_after_encoding: Optional[list] = None
        
        logger.info("Heart disease data preprocessor initialized")
    
    def load_raw_dataset(self, data_url: str) -> pd.DataFrame:
        """
        Load and perform initial cleaning of the heart disease dataset.
        
        Args:
            data_url: URL to the dataset
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Loading dataset from: {data_url}")
            
            # Load data with proper column names
            raw_data = pd.read_csv(
                data_url, 
                names=self.feature_column_names + [self.target_column_name],
                na_values='?'
            )
            
            logger.info(f"Dataset loaded successfully. Shape: {raw_data.shape}")
            
            # Basic data quality checks
            self._perform_data_quality_checks(raw_data)
            
            return raw_data
            
        except Exception as error:
            logger.error(f"Failed to load dataset: {error}")
            raise
    
    def _perform_data_quality_checks(self, dataframe: pd.DataFrame) -> None:
        """Perform basic data quality validation."""
        
        logger.info("Performing data quality checks...")
        
        # Check for missing values
        missing_data_summary = dataframe.isnull().sum()
        if missing_data_summary.any():
            logger.warning(f"Missing values detected:\n{missing_data_summary[missing_data_summary > 0]}")
        
        # Check data types
        logger.info(f"Data types:\n{dataframe.dtypes}")
        
        # Check target distribution
        target_distribution = dataframe[self.target_column_name].value_counts()
        logger.info(f"Target distribution:\n{target_distribution}")
    
    def create_preprocessing_pipeline(self, dataframe: pd.DataFrame) -> ColumnTransformer:
        """
        Create comprehensive preprocessing pipeline.
        
        Args:
            dataframe: Input DataFrame for pipeline fitting
            
        Returns:
            Fitted ColumnTransformer pipeline
        """
        
        logger.info("Creating preprocessing pipeline...")
        
        # Separate features from target
        features_df = dataframe.drop(self.target_column_name, axis=1)
        
        # Identify numeric and categorical features
        numeric_feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_feature_names = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        logger.info(f"Numeric features: {numeric_feature_names}")
        logger.info(f"Categorical features: {categorical_feature_names}")
        
        # Create preprocessing pipelines for different feature types
        numeric_preprocessing_pipeline = Pipeline([
            ('missing_value_imputer', SimpleImputer(strategy='median')),
            ('feature_scaler', StandardScaler())
        ])
        
        categorical_preprocessing_pipeline = Pipeline([
            ('missing_value_imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessing_pipeline = ColumnTransformer([
            ('numeric_features', numeric_preprocessing_pipeline, numeric_feature_names),
            ('categorical_features', categorical_preprocessing_pipeline, categorical_feature_names)
        ])
        
        logger.info("Preprocessing pipeline created successfully")
        return self.preprocessing_pipeline
    
    def prepare_training_data(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            dataframe: Raw dataset
            
        Returns:
            Tuple of (processed_features, binary_target, feature_names)
        """
        
        logger.info("Preparing training data...")
        
        # Separate features and target
        feature_data = dataframe.drop(self.target_column_name, axis=1)
        target_data = dataframe[self.target_column_name]
        
        # Convert multi-class target to binary (0: no disease, 1: disease)
        binary_target = (target_data > 0).astype(int)
        
        # Create and fit preprocessing pipeline
        preprocessing_pipeline = self.create_preprocessing_pipeline(dataframe)
        processed_features = preprocessing_pipeline.fit_transform(feature_data)
        
        # Get feature names after preprocessing
        self.feature_names_after_encoding = self._extract_feature_names_after_encoding(
            preprocessing_pipeline, feature_data
        )
        
        logger.info(f"Training data prepared. Features shape: {processed_features.shape}")
        logger.info(f"Target distribution: {np.bincount(binary_target)}")
        
        return processed_features, binary_target, self.feature_names_after_encoding
    
    def _extract_feature_names_after_encoding(
        self, 
        pipeline: ColumnTransformer, 
        original_features: pd.DataFrame
    ) -> list:
        """Extract feature names after one-hot encoding."""
        
        try:
            feature_names = pipeline.get_feature_names_out()
            return list(feature_names)
        except Exception as error:
            logger.warning(f"Could not extract feature names: {error}")
            return [f"feature_{i}" for i in range(pipeline.transform(original_features).shape[1])]
    
    def transform_single_prediction_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        Transform single patient data for prediction.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Processed feature array
        """
        
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not fitted. Call prepare_training_data first.")
        
        # Convert to DataFrame with proper column names
        input_dataframe = pd.DataFrame([patient_data])
        
        # Apply preprocessing
        processed_input = self.preprocessing_pipeline.transform(input_dataframe)
        
        return processed_input
    
    def get_feature_importance_mapping(self) -> Dict[str, int]:
        """Get mapping of feature names to indices after preprocessing."""
        
        if self.feature_names_after_encoding is None:
            raise ValueError("Feature names not available. Run data preparation first.")
        
        return {
            feature_name: index 
            for index, feature_name in enumerate(self.feature_names_after_encoding)
        }
