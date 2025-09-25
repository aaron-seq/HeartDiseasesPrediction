"""
Logging configuration for the application.
"""

import sys
from loguru import logger
from .config import app_settings


def configure_logging():
    """Configure application logging."""
    
    # Remove default handler
    logger.remove()
    
    # Add custom handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<white>{message}</white>",
        level=app_settings.log_level
    )
    
    # Add file handler for production
    logger.add(
        "logs/heart_disease_api.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
