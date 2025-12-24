"""
Application configuration settings.
Handles environment variables and application settings.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class ApplicationSettings(BaseSettings):
    """Application configuration settings."""

    # API Settings
    api_title: str = "Heart Disease Prediction API"
    api_description: str = "Advanced ML-powered heart disease risk assessment"
    api_version: str = "2.0.0"
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # CORS Settings
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "https://your-vercel-app.vercel.app",
    ]

    # ML Model Settings
    model_path: str = Field(default="models/saved_models", env="MODEL_PATH")
    data_url: str = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Database Settings
    database_url: str = Field(
        default="sqlite:///./heart_disease_predictions.db", env="DATABASE_URL"
    )

    # MLflow Settings
    mlflow_tracking_uri: str = Field(default="./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = "heart-disease-prediction"

    # Security Settings
    secret_key: str = Field(
        default="your-secret-key-change-in-production", env="SECRET_KEY"
    )

    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
app_settings = ApplicationSettings()
