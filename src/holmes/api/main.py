"""FastAPI application for Holmes AI transaction categorization."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger
import numpy as np

from ..ingestion.schema import Transaction
from ..preprocessing.cleaner import MerchantCleaner
from ..preprocessing.enrichment import FeatureExtractor
from ..encoding.semantic_encoder import SemanticEncoder
from ..classification.classifier import TransactionClassifier
from ..monitoring.drift_detector import PerformanceMonitor
from ..feedback.learning import FeedbackCollector


# Initialize FastAPI app
app = FastAPI(
    title="Holmes AI",
    description="AI-native transaction categorization engine with enterprise-grade accuracy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized on startup)
merchant_cleaner = None
feature_extractor = None
semantic_encoder = None
classifier = None
performance_monitor = None
feedback_collector = None


# Request/Response models
class CategorizeRequest(BaseModel):
    """Request model for single transaction categorization."""

    transaction_id: str
    merchant: str
    amount: float
    date: datetime
    channel: Optional[str] = None
    location: Optional[str] = None
    currency: str = "USD"

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN123456",
                "merchant": "STARBUCKS STORE #12345",
                "amount": 5.75,
                "date": "2025-11-12T08:30:00",
                "channel": "in-store",
                "location": "Seattle, WA"
            }
        }


class BatchCategorizeRequest(BaseModel):
    """Request model for batch categorization."""

    transactions: List[CategorizeRequest]


class CategorizeResponse(BaseModel):
    """Response model for categorization."""

    transaction_id: str
    merchant: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""

    transaction_id: str
    merchant: str
    predicted_category: str
    corrected_category: str
    confidence: float
    user_id: Optional[str] = None
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    encoder_loaded: bool
    timestamp: datetime


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global merchant_cleaner, feature_extractor, semantic_encoder, classifier
    global performance_monitor, feedback_collector

    logger.info("Starting Holmes AI API...")

    try:
        # Initialize components
        merchant_cleaner = MerchantCleaner()
        feature_extractor = FeatureExtractor()
        semantic_encoder = SemanticEncoder()
        classifier = TransactionClassifier()
        performance_monitor = PerformanceMonitor()
        feedback_collector = FeedbackCollector()

        # Try to load pre-trained model
        try:
            classifier.load("models/classifier.pkl")
            logger.info("Pre-trained classifier loaded")
        except FileNotFoundError:
            logger.warning("No pre-trained classifier found. Train model first.")

        logger.info("Holmes AI API started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "Holmes AI",
        "version": "1.0.0",
        "description": "AI-native transaction categorization engine"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier.is_fitted if classifier else False,
        encoder_loaded=semantic_encoder is not None,
        timestamp=datetime.now()
    )


@app.post("/categorize", response_model=CategorizeResponse)
async def categorize_transaction(
    request: CategorizeRequest,
    background_tasks: BackgroundTasks
):
    """
    Categorize a single transaction.

    Returns predicted category, confidence, and probabilities.
    Latency: <200ms
    """
    if not classifier or not classifier.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first."
        )

    start_time = datetime.now()

    try:
        # 1. Clean merchant description
        cleaned = merchant_cleaner.clean(request.merchant)

        # 2. Extract numeric features
        features = feature_extractor.extract({
            "merchant": request.merchant,
            "amount": request.amount,
            "date": request.date,
            "location": request.location
        })

        # 3. Generate semantic embedding
        embedding = semantic_encoder.encode(cleaned["cleaned"])

        # 4. Combine features
        combined_features = np.concatenate([embedding.flatten(), features])

        # 5. Classify
        prediction = classifier.predict_single(combined_features)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log for monitoring
        background_tasks.add_task(
            performance_monitor.log_prediction,
            category=prediction["category"],
            confidence=prediction["confidence"]
        )

        logger.info(
            f"Categorized: {request.merchant} -> {prediction['category']} "
            f"({prediction['confidence']:.2f}) in {processing_time:.0f}ms"
        )

        return CategorizeResponse(
            transaction_id=request.transaction_id,
            merchant=request.merchant,
            category=prediction["category"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/categorize/batch", response_model=List[CategorizeResponse])
async def categorize_batch(
    request: BatchCategorizeRequest,
    background_tasks: BackgroundTasks
):
    """
    Categorize a batch of transactions.

    Optimized for throughput: processes 10-20M records/month.
    """
    if not classifier or not classifier.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first."
        )

    results = []

    for txn in request.transactions:
        try:
            result = await categorize_transaction(txn, background_tasks)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to categorize transaction {txn.transaction_id}: {e}")
            # Continue with other transactions

    logger.info(f"Batch categorization complete: {len(results)}/{len(request.transactions)}")

    return results


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a prediction.

    Used for continuous learning and model improvement.
    """
    try:
        feedback_collector.add_feedback(
            transaction_id=request.transaction_id,
            merchant=request.merchant,
            predicted_category=request.predicted_category,
            corrected_category=request.corrected_category,
            confidence=request.confidence,
            user_id=request.user_id,
            notes=request.notes
        )

        return {
            "status": "success",
            "message": "Feedback recorded successfully"
        }

    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics."""
    try:
        stats = feedback_collector.get_correction_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/stats")
async def get_monitoring_stats():
    """Get performance monitoring statistics."""
    try:
        stats = performance_monitor.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get monitoring stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """Get model information and feature importance."""
    if not classifier or not classifier.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_importance = classifier.get_feature_importance(top_k=20)

        return {
            "model_type": "LightGBM",
            "is_fitted": classifier.is_fitted,
            "n_classes": len(classifier.label_encoder.classes_),
            "classes": classifier.label_encoder.classes_.tolist(),
            "feature_importance": [
                {"feature": name, "importance": float(imp)}
                for name, imp in feature_importance
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/taxonomy")
async def get_taxonomy():
    """Get category taxonomy."""
    try:
        import json
        from pathlib import Path

        taxonomy_path = Path("config/taxonomy.json")
        if not taxonomy_path.exists():
            raise HTTPException(status_code=404, detail="Taxonomy not found")

        with open(taxonomy_path, "r") as f:
            taxonomy = json.load(f)

        return taxonomy

    except Exception as e:
        logger.error(f"Failed to get taxonomy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
