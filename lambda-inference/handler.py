"""
handler.py
==========
Lambda entry point for fashion image classification.

Receives a JSON payload:
    {
        "image_b64": "<base64-encoded JPEG or PNG>",
        "image_size": 224          # optional, default 224
    }

Returns:
    {
        "predicted_class": "blazer",
        "confidence": 0.923,
        "all_scores": {"biker": 0.01, "blazer": 0.92, ...},
        "timing": {
            "cold_start_ms": 4200,   # only present on first invocation
            "inference_ms": 38
        }
    }

Environment variables required:
    MODEL_BUCKET      — S3 bucket name (e.g. "my-fashion-models")
    MODEL_KEY_CKPT    — S3 key for checkpoint (e.g. "jackets/01/best_weights.pt")
    MODEL_KEY_NORM    — S3 key for normalization (e.g. "jackets/01/normalization.npy")
    MODEL_CLASSES     — comma-separated class names in training order
                        (e.g. "biker,blazer,bomber,fur jacket,parka")
    MODEL_IMAGE_SIZE  — optional, default "224"
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import boto3
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Module-level globals — persist across warm invocations
# ---------------------------------------------------------------------------
_model: nn.Module | None = None
_normalization: np.ndarray | None = None
_classes: list[str] | None = None
_image_size: int | None = None
_cold_start_ms: float | None = None   # set once, reported on first invocation only

CACHE_DIR = Path("/tmp/model_cache")


# ---------------------------------------------------------------------------
# Model definition — plain PyTorch, no Lightning dependency
# ---------------------------------------------------------------------------

def build_resnet50_head(num_classes: int) -> nn.Module:
    """
    Reconstructs the same architecture used in FashionClassifier (resnet50 backbone).
    The checkpoint was saved by Lightning, but we load state_dict only —
    no Lightning or Hydra needed at inference time.
    """
    model = resnet50(weights=None)
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_weights_from_lightning_ckpt(ckpt_path: Path, model: nn.Module) -> nn.Module:
    """
    Lightning checkpoints wrap the state_dict under 'state_dict' key,
    and prefix all keys with 'model.' (for ResNet) or 'encoder.'/'head.' (for CLIP).
    We strip the prefix and load into plain PyTorch model.
    """
    # map_location="cpu" is essential: Lambda has no GPU, and checkpoints
    # saved on CUDA will fail without this explicit remap.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"]

    # Strip 'model.' prefix that Lightning adds
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        # skip head.*, loss_fn.*, train_acc.*, val_acc.* etc.

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys (ignored): {unexpected}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# S3 download with /tmp caching
# ---------------------------------------------------------------------------

def download_from_s3(bucket: str, key: str, local_path: Path) -> None:
    if local_path.exists():
        logger.info(f"Cache hit: {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading s3://{bucket}/{key} → {local_path}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
    logger.info(f"Downloaded {local_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Image preprocessing — mirrors preprocess.py exactly
# ---------------------------------------------------------------------------

def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w == h:
        return img
    size = max(w, h)
    result = np.full((size, size, 3), 255, dtype=img.dtype)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    result[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return result


def preprocess_image(image_bytes: bytes, image_size: int, normalization: np.ndarray) -> torch.Tensor:
    """
    bytes → (1, 3, H, W) float32 tensor, normalized to training stats.
    Mirrors preprocess.py: pad to square → resize → normalize → CHW.
    """
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: try PIL
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(pil)[:, :, ::-1]  # RGB → BGR for consistency, then back
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = pad_to_square(img)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    mean = normalization[0]   # (3,)
    std  = normalization[1]   # (3,)
    img  = (img - mean) / (std + 1e-8)

    # HWC → CHW → batch dim
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


# ---------------------------------------------------------------------------
# Cold start initialisation
# ---------------------------------------------------------------------------

def initialise() -> None:
    """
    Load model + normalization from S3 into module globals.
    Called once per container lifetime — measures cold start time.
    """
    global _model, _normalization, _classes, _image_size, _cold_start_ms

    t0 = time.perf_counter()

    bucket    = os.environ["MODEL_BUCKET"]
    ckpt_key  = os.environ["MODEL_KEY_CKPT"]
    norm_key  = os.environ["MODEL_KEY_NORM"]
    classes   = [c.strip() for c in os.environ["MODEL_CLASSES"].split(",")]
    img_size  = int(os.environ.get("MODEL_IMAGE_SIZE", "224"))

    ckpt_path = CACHE_DIR / "best_weights.pt"
    norm_path = CACHE_DIR / "normalization.npy"

    download_from_s3(bucket, ckpt_key, ckpt_path)
    download_from_s3(bucket, norm_key, norm_path)

    norm = np.load(str(norm_path))  # shape (2, 3)

    model = build_resnet50_head(num_classes=len(classes))
    model = load_weights_from_lightning_ckpt(ckpt_path, model)

    _model        = model
    _normalization = norm
    _classes      = classes
    _image_size   = img_size
    _cold_start_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"Initialised — classes={classes}  image_size={img_size}  "
        f"cold_start={_cold_start_ms:.0f}ms"
    )


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    # Lazy initialisation — runs once per container
    is_cold = _model is None
    if is_cold:
        initialise()

    # ── Parse input ──────────────────────────────────────────────────────
    body = event if "image_b64" in event else json.loads(event.get("body", "{}"))

    image_b64  = body.get("image_b64")
    image_size = int(body.get("image_size", _image_size))

    if not image_b64:
        return {"statusCode": 400, "body": json.dumps({"error": "image_b64 required"})}

    image_bytes = base64.b64decode(image_b64)

    # ── Inference ────────────────────────────────────────────────────────
    t_infer = time.perf_counter()

    tensor = preprocess_image(image_bytes, image_size, _normalization)

    with torch.no_grad():
        logits = _model(tensor)                        # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]       # (num_classes,)

    pred_idx    = int(probs.argmax().item())
    confidence  = float(probs[pred_idx].item())
    all_scores  = {cls: round(float(p), 4) for cls, p in zip(_classes, probs.tolist())}

    inference_ms = (time.perf_counter() - t_infer) * 1000

    # ── Response ─────────────────────────────────────────────────────────
    timing: dict = {"inference_ms": round(inference_ms, 1)}
    if is_cold:
        timing["cold_start_ms"] = round(_cold_start_ms, 1)

    result = {
        "predicted_class": _classes[pred_idx],
        "confidence":      round(confidence, 4),
        "all_scores":      all_scores,
        "timing":          timing,
    }

    logger.info(f"Result: {result}")
    return {"statusCode": 200, "body": json.dumps(result)}
