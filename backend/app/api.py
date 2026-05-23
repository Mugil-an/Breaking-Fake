"""FastAPI service for Breaking-Fake model."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import urllib.error
import urllib.request

import backend.app.auth as auth
import logging

from shared.config import settings
from logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="AI-Generated vs Real Image Detection API"
)

# Allow CORS in development so the frontend (Vite/dev server) can call the API
if settings.DEBUG:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class PredictionResult(BaseModel):
    """Single prediction result."""
    class_label: int
    class_name: str
    confidence: float
    probabilities: dict

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_service_url": settings.MODEL_SERVICE_URL,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(auth.get_api_key)):
    """Predict on a single image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Prediction result with class, confidence, and probabilities
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty upload")

        request = urllib.request.Request(
            url=f"{settings.MODEL_SERVICE_URL.rstrip('/')}/predict",
            data=content,
            method="POST",
            headers={
                "Content-Type": file.content_type or "application/octet-stream",
                "X-Filename": file.filename or "upload.jpg",
            },
        )

        with urllib.request.urlopen(request, timeout=settings.MODEL_SERVICE_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return PredictionResult(
            class_label=payload["class"],
            class_name=payload["class_name"],
            confidence=payload["confidence"],
            probabilities=payload["probabilities"],
        )

    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        logger.error(f"Prediction service error: {e.code} {detail}")
        raise HTTPException(status_code=502, detail="Model service returned an error")
    except urllib.error.URLError as e:
        logger.error(f"Model service unavailable: {e}")
        raise HTTPException(status_code=503, detail="Model service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "model_name": settings.MODEL_NAME,
        "num_classes": settings.NUM_CLASSES,
        "image_size": settings.IMAGE_SIZE,
        "classes": settings.CLASS_NAMES,
        "checkpoint": settings.CHECKPOINT_NAME,
        "device": settings.DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG
    )