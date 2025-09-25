"""
Advanced heart disease prediction model with enhanced neural network architecture.
Includes model training, prediction, and explainability features.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import shap
import joblib
import mlflow
import mlflow.tensorflow
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from loguru import logger
import json
from datetime import datetime

from ..utils.data_preprocessing import HeartDiseaseDataPreprocessor
from ..core.config import app_settings


class EnhancedHeartDiseasePredictor:
    """
    Advanced neural network model for heart disease prediction.
    Includes comprehensive training, evaluation, and explanation capabilities.
    """
    
    def __init__(self):
        """Initialize the heart disease predictor."""
        self.model: Optional[keras.Model] = None
        self.data_preprocessor = HeartDiseaseDataPreprocessor()
        self.model_metadata = {}
        self.shap_explainer: Optional[shap.Explainer] = None
        
        # Model architecture parameters
        self.model_architecture_config = {
            'dense_layer_1_units': 128,
            'dense_layer_2_units': 64,
            'dense_layer_3_units': 32,
            'dropout_rate': 0.3,
            'batch_normalization': True,
            'activation_function': 'relu',
            'kernel_initializer': 'he_normal'
        }
        
        # Training parameters
        self.training_config = {
            'epochs': 150,
            'batch_size': 32,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10
        }
        
        logger.info("Enhanced heart disease predictor initialized")
    
    def _create_advanced_neural_network(self, input_feature_count: int) -> keras.Model:
        """
        Create advanced neural network architecture with modern techniques.
        
        Args:
            input_feature_count: Number of input features
            
        Returns:
            Compiled Keras model
        """
        
        logger.info(f"Creating neural network with {input_feature_count} input features")
        
        # Input layer
        input_layer = keras.Input(shape=(input_feature_count,), name='patient_features')
        
        # First dense block
        x = layers.Dense(
            self.model_architecture_config['dense_layer_1_units'],
            kernel_initializer=self.model_architecture_config['kernel_initializer'],
            name='dense_layer_1'
        )(input_layer)
        
        if self.model_architecture_config['batch_normalization']:
            x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        x = layers.Activation(
            self.model_architecture_config['activation_function'], 
            name='activation_1'
        )(x)
        x = layers.Dropout(
            self.model_architecture_config['dropout_rate'], 
            name='dropout_1'
        )(x)
        
        # Second dense block
        x = layers.Dense(
            self.model_architecture_config['dense_layer_2_units'],
            kernel_initializer=self.model_architecture_config['kernel_initializer'],
            name='dense_layer_2'
        )(x)
        
        if self.model_architecture_config['batch_normalization']:
            x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        x = layers.Activation(
            self.model_architecture_config['activation_function'], 
            name='activation_2'
        )(x)
        x = layers.Dropout(
            self.model_architecture_config['dropout_rate'], 
            name='dropout_2'
        )(x)
        
        # Third dense block
        x = layers.Dense(
            self.model_architecture_config['dense_layer_3_units'],
            kernel_initializer=self.model_architecture_config['kernel_initializer'],
            name='dense_layer_3'
        )(x)
        
        if self.model_architecture_config['batch_normalization']:
            x = layers.BatchNormalization(name='batch_norm_3')(x)
        
        x = layers.Activation(
            self.model_architecture_config['activation_function'], 
            name='activation_3'
        )(x)
        x = layers.Dropout(
            self.model_architecture_config['dropout_rate'], 
            name='dropout_3'
        )(x)
        
        # Output layer for binary classification
        output_layer = layers.Dense(
            1, 
            activation='sigmoid', 
            name='heart_disease_prediction'
        )(x)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output_layer, name='HeartDiseasePredictor')
        
        # Compile model with advanced optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        logger.info("Neural network architecture created and compiled")
        logger.info(f"Model summary:\n{model.summary()}")
        
        return model
    
    def train_model(self, dataset_url: str = None) -> Dict[str, Any]:
        """
        Train the heart disease prediction model with comprehensive evaluation.
        
        Args:
            dataset_url: URL to training dataset
            
        Returns:
            Training metrics and model information
        """
        
        logger.info("Starting model training process...")
        
        # Use default dataset URL if not provided
        if dataset_url is None:
            dataset_url = app_settings.data_url
        
        # Start MLflow experiment tracking
        mlflow.set_tracking_uri(app_settings.mlflow_tracking_uri)
        mlflow.set_experiment(app_settings.mlflow_experiment_name)
        
        with mlflow.start_run():
            # Load and prepare data
            raw_dataset = self.data_preprocessor.load_raw_dataset(dataset_url)
            processed_features, binary_target, feature_names = self.data_preprocessor.prepare_training_data(raw_dataset)
            
            # Split data for training and testing
            (features_train, features_test, 
             target_train, target_test) = train_test_split(
                processed_features, binary_target,
                test_size=0.2, 
                random_state=42, 
                stratify=binary_target
            )
            
            logger.info(f"Training set size: {features_train.shape[0]}")
            logger.info(f"Test set size: {features_test.shape[0]}")
            
            # Create model
            self.model = self._create_advanced_neural_network(features_train.shape[1])
            
            # Define callbacks for training
            training_callbacks = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.training_config['reduce_lr_patience'],
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    filepath='best_heart_disease_model.h5',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Log hyperparameters to MLflow
            mlflow.log_params(self.model_architecture_config)
            mlflow.log_params(self.training_config)
            
            # Train model
            logger.info("Starting model training...")
            training_history = self.model.fit(
                features_train, target_train,
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                validation_split=self.training_config['validation_split'],
                callbacks=training_callbacks,
                verbose=1
            )
            
            # Evaluate model performance
            test_predictions = self.model.predict(features_test)
            binary_predictions = (test_predictions > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(target_test, binary_predictions)
            roc_auc = roc_auc_score(target_test, test_predictions)
            
            # Log metrics to MLflow
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_roc_auc", roc_auc)
            
            # Log model to MLflow
            mlflow.tensorflow.log_model(self.model, "heart_disease_model")
            
            # Save model metadata
            self.model_metadata = {
                'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_date': datetime.now().isoformat(),
                'test_accuracy': accuracy,
                'test_roc_auc': roc_auc,
                'feature_count': features_train.shape[1],
                'training_samples': features_train.shape[0],
                'feature_names': feature_names
            }
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Initialize SHAP explainer for model interpretability
            self._initialize_shap_explainer(features_train)
            
            # Save model and preprocessor
            self.save_model()
            
            return {
                'training_history': training_history.history,
                'test_metrics': {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'classification_report': classification_report(target_test, binary_predictions, output_dict=True)
                },
                'model_metadata': self.model_metadata
            }
    
    def _initialize_shap_explainer(self, training_features: np.ndarray) -> None:
        """Initialize SHAP explainer for model interpretability."""
        
        logger.info("Initializing SHAP explainer...")
        
        try:
            # Use a sample of training data for SHAP background
            background_sample = training_features[:100]  # Use first 100 samples as background
            
            # Create SHAP explainer
            self.shap_explainer = shap.DeepExplainer(self.model, background_sample)
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as error:
            logger.warning(f"Failed to initialize SHAP explainer: {error}")
            self.shap_explainer = None
    
    def predict_heart_disease_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict heart disease risk for a single patient.
        
        Args:
            patient_data: Patient health information
            
        Returns:
            Prediction results with risk assessment
        """
        
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        logger.info("Making heart disease risk prediction...")
        
        # Preprocess patient data
        processed_input = self.data_preprocessor.transform_single_prediction_input(patient_data)
        
        # Make prediction
        risk_probability = float(self.model.predict(processed_input)[0][0])
        has_risk = risk_probability > 0.5
        
        # Categorize risk level
        if risk_probability < 0.3:
            risk_level = "Low Risk"
        elif risk_probability < 0.7:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
        
        # Calculate confidence (distance from decision boundary)
        confidence_score = max(risk_probability, 1 - risk_probability)
        
        prediction_result = {
            'has_heart_disease_risk': has_risk,
            'risk_probability': risk_probability,
            'risk_level': risk_level,
            'confidence_score': confidence_score,
            'model_version': self.model_metadata.get('model_version', 'unknown')
        }
        
        # Add SHAP explanations if available
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(processed_input)
                feature_importance = self._interpret_shap_values(shap_values[0], patient_data)
                prediction_result.update(feature_importance)
            except Exception as error:
                logger.warning(f"Failed to generate SHAP explanations: {error}")
        
        logger.info(f"Prediction completed. Risk probability: {risk_probability:.4f}")
        
        return prediction_result
    
    def _interpret_shap_values(self, shap_values: np.ndarray, patient_data: Dict[str, Any]) -> Dict:
        """Interpret SHAP values to provide meaningful explanations."""
        
        # Get feature names
        feature_names = self.model_metadata.get('feature_names', [])
        
        if len(feature_names) != len(shap_values):
            logger.warning("Feature names and SHAP values length mismatch")
            return {}
        
        # Create feature importance ranking
        feature_importance_pairs = list(zip(feature_names, shap_values))
        feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Identify top risk factors (positive SHAP values)
        risk_factors = [
            name for name, value in feature_importance_pairs[:5] 
            if value > 0
        ]
        
        # Identify protective factors (negative SHAP values)
        protective_factors = [
            name for name, value in feature_importance_pairs[:5] 
            if value < 0
        ]
        
        return {
            'primary_risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'feature_importance_scores': dict(feature_importance_pairs[:10])
        }
    
    def save_model(self, model_directory: str = None) -> None:
        """Save trained model and associated components."""
        
        if model_directory is None:
            model_directory = app_settings.model_path
        
        model_path = Path(model_directory)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_file = model_path / "heart_disease_model.h5"
        self.model.save(model_file)
        
        # Save preprocessor
        preprocessor_file = model_path / "data_preprocessor.pkl"
        joblib.dump(self.data_preprocessor.preprocessing_pipeline, preprocessor_file)
        
        # Save metadata
        metadata_file = model_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_directory}")
    
    def load_model(self, model_directory: str = None) -> None:
        """Load trained model and associated components."""
        
        if model_directory is None:
            model_directory = app_settings.model_path
        
        model_path = Path(model_directory)
        
        # Load Keras model
        model_file = model_path / "heart_disease_model.h5"
        self.model = keras.models.load_model(model_file)
        
        # Load preprocessor
        preprocessor_file = model_path / "data_preprocessor.pkl"
        self.data_preprocessor.preprocessing_pipeline = joblib.load(preprocessor_file)
        
        # Load metadata
        metadata_file = model_path / "model_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
        
        logger.info(f"Model loaded from {model_directory}")
