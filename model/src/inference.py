"""Production inference service for Breaking-Fake model."""
from pathlib import Path
from typing import Dict, List
import logging

from shared.config import settings

logger = logging.getLogger(__name__)


class BreakingFakeModel:
    """Production model wrapper for Breaking-Fake detection."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        """Initialize model.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on ('cuda' or 'cpu')
        """
        self.device = device or settings.DEVICE
        self.torch_device = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else settings.checkpoint_path
        self.model = None
        self.transform = None

        self._load_model()
        self._setup_transforms()
        logger.info(f"Model initialized on device: {self.device}")

    def _load_model(self):
        """Load model from checkpoint."""
        import torch
        import timm

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            self.model = timm.create_model(
                settings.MODEL_NAME,
                pretrained=False,
                num_classes=settings.NUM_CLASSES,
            )

            if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
                map_location = torch.device(self.device)
            else:
                map_location = torch.device("cpu")

            checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            try:
                model_pos = None
                if hasattr(self.model, "pos_embed"):
                    model_pos = self.model.pos_embed

                if model_pos is not None and "pos_embed" in state_dict:
                    ck_pos = state_dict["pos_embed"]
                    if ck_pos.shape != model_pos.shape:
                        import torch.nn.functional as F

                        _, old_num, dim = ck_pos.shape
                        _, new_num, _ = model_pos.shape

                        old_grid = int((old_num - 1) ** 0.5)
                        new_grid = int((new_num - 1) ** 0.5)

                        if old_grid * old_grid == (old_num - 1) and new_grid * new_grid == (new_num - 1):
                            cls_token = ck_pos[:, :1, :]
                            pos_tokens = ck_pos[:, 1:, :].reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                            pos_tokens = F.interpolate(pos_tokens, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
                            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
                            state_dict["pos_embed"] = torch.cat((cls_token, pos_tokens), dim=1)
                            logger.info(f"Interpolated pos_embed from {old_grid}x{old_grid} to {new_grid}x{new_grid}")
            except Exception as e:
                logger.warning(f"Could not interpolate pos_embed: {e}")

            self.model.load_state_dict(state_dict)
            self.torch_device = map_location
            self.model = self.model.to(self.torch_device)
            self.model.eval()

            logger.info(f"Model loaded from {self.checkpoint_path}")
            if "val_acc" in checkpoint:
                logger.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _setup_transforms(self):
        """Setup image transforms."""
        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_image(self, image_path: str) -> Dict:
        """Predict on a single image."""
        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.torch_device)

            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = logits.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()

            return {
                "class": pred_class,
                "class_name": settings.CLASS_NAMES[pred_class],
                "confidence": float(confidence),
                "probabilities": {
                    "AI-generated": float(probs[0, 0].item()),
                    "Real": float(probs[0, 1].item()),
                },
                "image_path": str(image_path),
            }
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            raise

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict on multiple images."""
        results = []

        for image_path in image_paths:
            try:
                results.append(self.predict_image(image_path))
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({"image_path": str(image_path), "error": str(e)})

        return results

    def reload_checkpoint(self, checkpoint_path: str = None):
        """Reload model from checkpoint."""
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)

        self._load_model()
        logger.info(f"Model reloaded from {self.checkpoint_path}")


_model_instance = None


def get_model(checkpoint_path: str = None) -> BreakingFakeModel:
    """Get or create model instance."""
    global _model_instance

    if _model_instance is None:
        _model_instance = BreakingFakeModel(checkpoint_path=checkpoint_path)

    return _model_instance


def reload_model(checkpoint_path: str = None):
    """Reload the model instance."""
    global _model_instance
    _model_instance = None
    return get_model(checkpoint_path=checkpoint_path)
