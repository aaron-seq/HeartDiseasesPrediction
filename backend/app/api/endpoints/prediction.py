"""
API endpoints for heart disease prediction.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from datetime import datetime
from loguru import logger

from ...models.schemas import (
    PredictionRequest, 
    HeartDiseaseRiskPrediction, 
    PatientHealthData,
    ModelHealthStatus
)
from ...services.ml_service import ml_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=HeartDiseaseRiskPrediction,
    summary="Predict Heart Disease Risk",
    description="Analyze patient health data and provide heart disease risk assessment with personalized recommendations."
)
async def predict_heart_disease_risk(
    prediction_request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> HeartDiseaseRiskPrediction:
    """
    Predict heart disease risk for a patient.
    
    This endpoint analyzes comprehensive patient health data using an advanced
    neural network model to assess heart disease risk. The response includes:
    
    - Binary risk prediction (at risk / not at risk)
    - Risk probability score (0-1)
    - Risk level categorization (Low/Moderate/High)
    - Model confidence score
    - Primary risk factors identified
    - Protective factors present
    - Personalized lifestyle recommendations
    - Medical consultation recommendations
    
    Args:
        prediction_request: Patient health data and prediction options
        background_tasks: Background task handler for logging
        
    Returns:
        Comprehensive heart disease risk assessment
        
    Raises:
        HTTPException: If prediction fails or model is unavailable
    """
    
    try:
        logger.info("Received heart disease prediction request")
        
        # Validate service availability
        if not ml_service.is_model_loaded:
            raise HTTPException(
                status_code=503,
                detail="ML model not available. Service is initializing."
            )
        
        # Make prediction
        prediction_result = await ml_service.predict_heart_disease(
            prediction_request.patient_data
        )
        
        # Log prediction in background (async)
        background_tasks.add_task(
            log_prediction_request,
            prediction_request.patient_data,
            prediction_result
        )
        
        logger.info("Heart disease prediction completed successfully")
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Prediction endpoint failed: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


@router.get(
    "/model-status",
    response_model=ModelHealthStatus,
    summary="Get Model Health Status",
    description="Retrieve current model health, performance metrics, and version information."
)
async def get_model_health_status() -> ModelHealthStatus:
    """
    Get comprehensive model health and performance information.
    
    Returns:
        Model health status including version, accuracy, and performance metrics
    """
    
    try:
        model_status = ml_service.get_model_health_status()
        return model_status
        
    except Exception as error:
        logger.error(f"Failed to get model status: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model status"
        )


@router.post(
    "/retrain-model",
    summary="Retrain Model",
    description="Trigger model retraining with latest data (Admin endpoint)."
)
async def retrain_model(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Retrain the heart disease prediction model.
    
    This endpoint triggers model retraining in the background.
    Note: In production, this should be secured with proper authentication.
    
    Args:
        background_tasks: Background task handler
        
    Returns:
        Status message about retraining initiation
    """
    
    try:
        # Add retraining task to background
        background_tasks.add_task(retrain_model_background)
        
        return {
            "status": "success",
            "message": "Model retraining initiated in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as error:
        logger.error(f"Failed to initiate model retraining: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate model retraining"
        )


async def log_prediction_request(
    patient_data: PatientHealthData,
    prediction_result: HeartDiseaseRiskPrediction
) -> None:
    """Log prediction request for monitoring and analytics."""
    
    try:
        # In production, this would log to a database or analytics service
        logger.info(
            f"PREDICTION_LOG: "
            f"Age={patient_data.age}, "
            f"Sex={'M' if patient_data.is_male else 'F'}, "
            f"Risk={prediction_result.risk_probability:.3f}, "
            f"Level={prediction_result.risk_level}"
        )
        
    except Exception as error:
        logger.warning(f"Failed to log prediction request: {error}")


async def retrain_model_background() -> None:
    """Background task for model retraining."""
    
    try:
        logger.info("Starting background model retraining...")
        
        # Reinitialize the service (which will retrain if needed)
        await ml_service._train_new_model()
        
        logger.info("Model retraining completed successfully")
        
    except Exception as error:
        logger.error(f"Background model retraining failed: {error}")
