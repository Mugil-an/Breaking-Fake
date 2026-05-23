"""Production configuration for Breaking-Fake model."""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore"
    )
    
    # Project info
    PROJECT_NAME: str = "Breaking-Fake"
    PROJECT_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # Paths
    # Path to project root (this file lives in the project root)
    PROJECT_ROOT: Path = Path(__file__).parent
    MODEL_DIR: Path = PROJECT_ROOT / "model"
    ARTIFACTS_DIR: Path = MODEL_DIR / "artifacts"
    DATA_DIR: Path = MODEL_DIR / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Model configuration
    MODEL_NAME: str = "vit_base_patch16_384"
    CHECKPOINT_NAME: str = os.getenv("CHECKPOINT_NAME", "checkpoint_epoch_001_acc_0.9946.pt")
    NUM_CLASSES: int = 2  # AI-generated (0) vs Real (1)
    IMAGE_SIZE: int = 384
    
    # Device
    DEVICE: str = "cuda" if os.getenv("USE_CPU", "false").lower() != "true" else "cpu"
    
    # API configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    MODEL_SERVICE_URL: str = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:8001")
    MODEL_SERVICE_TIMEOUT: float = float(os.getenv("MODEL_SERVICE_TIMEOUT", "30"))
    # Auth & rate limiting
    API_KEY: str = os.getenv("API_KEY", "changeme")
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MIN: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    # Redis URL for background job queue (optional)
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    
    # Model inference settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = LOGS_DIR / "app.log"
    
    # Class names
    CLASS_NAMES: dict = {
        0: "AI-generated",
        1: "Real"
    }
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        self.LOGS_DIR.mkdir(exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def checkpoint_path(self) -> Path:
        """Get full path to checkpoint file."""
        return self.ARTIFACTS_DIR / self.CHECKPOINT_NAME
    
    @property
    def is_checkpoint_available(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_path.exists()


# Singleton instance
settings = Settings()
