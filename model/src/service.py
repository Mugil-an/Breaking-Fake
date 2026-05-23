"""Standalone inference API for the Breaking-Fake model."""
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from shared.config import settings
from model.src.inference import get_model

logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"{settings.PROJECT_NAME} Inference Service",
    version=settings.PROJECT_VERSION,
    description="Dedicated model service for AI-generated vs real image detection",
)

model = None


@app.on_event("startup")
async def startup_event():
    """Load the model once when the service starts."""
    global model

    if not settings.is_checkpoint_available:
        raise FileNotFoundError(f"Checkpoint not found: {settings.checkpoint_path}")

    model = get_model()
    logger.info("Inference model loaded in model service")


@app.get("/health")
async def health_check():
    """Model service health endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "checkpoint": str(settings.checkpoint_path),
    }


@app.post("/predict")
async def predict(request: Request):
    """Predict from raw image bytes forwarded by the backend."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image payload")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        return model.predict_image(tmp_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "model.src.service:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
    )