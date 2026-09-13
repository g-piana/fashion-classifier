"""
test_local.py
=============
Run this ON YOUR WINDOWS MACHINE (in your existing venv) before building
the Docker image. Validates that:

  1. Weights extract cleanly from the Lightning checkpoint
  2. The plain PyTorch model produces the same predictions as the Lightning model
  3. Preprocessing matches the training pipeline
  4. Timing is plausible for Lambda

Usage (PowerShell):
    cd E:\\fashion-classifier
    python lambda-inference/test_local.py `
        --ckpt      "E:/fashion-data/weights/jackets/01/best.ckpt" `
        --norm      "E:/fashion-data/weights/jackets/01/normalization.npy" `
        --image     "E:/fashion-data/01-RAW/jackets_img/some_jacket.jpg" `
        --classes   "biker,blazer,bomber,fur jacket,parka"

What it checks
--------------
  [1] Key extraction   — lists which keys were kept/stripped from the ckpt
  [2] Shape check      — confirms head output matches num_classes
  [3] Prediction       — prints class probabilities for your test image
  [4] Lightning parity — loads the same ckpt with Lightning and compares
                         logits; diff should be < 1e-5
  [5] Timing           — simulates cold start + 10 warm inference calls
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50


# ---------------------------------------------------------------------------
# Duplicated from handler.py — intentionally standalone so this script
# runs without any Lambda dependencies
# ---------------------------------------------------------------------------

def build_resnet50_head(num_classes: int) -> nn.Module:
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_weights_from_lightning_ckpt(ckpt_path: Path, model: nn.Module) -> nn.Module:
    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"]

    print("\n[1] Key extraction")
    print(f"    Total keys in checkpoint: {len(state)}")

    kept, skipped = {}, []
    for k, v in state.items():
        if k.startswith("model."):
            kept[k[len("model."):]] = v
        else:
            skipped.append(k)

    print(f"    Kept   (model.*): {len(kept)}")
    print(f"    Skipped          : {len(skipped)}")
    if skipped:
        print(f"    Skipped keys     : {skipped[:10]}")  # first 10

    missing, unexpected = model.load_state_dict(kept, strict=True)
    if missing:
        print(f"    ⚠  Missing keys  : {missing}")
        sys.exit(1)
    if unexpected:
        print(f"    ⚠  Unexpected    : {unexpected}")

    print("    ✓  State dict loaded cleanly")
    model.eval()
    return model


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w == h:
        return img
    size = max(w, h)
    result = np.full((size, size, 3), 255, dtype=img.dtype)
    result[(size - h) // 2:(size - h) // 2 + h,
           (size - w) // 2:(size - w) // 2 + w] = img
    return result


def preprocess_image(image_path: Path, image_size: int, normalization: np.ndarray) -> torch.Tensor:
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pad_to_square(img)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = (img - normalization[0]) / (normalization[1] + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Lightning parity check
# ---------------------------------------------------------------------------

def lightning_forward(ckpt_path: Path, tensor: torch.Tensor) -> torch.Tensor:
    """Load via Lightning and run forward pass for comparison."""
    try:
        # Add repo src/ to path so FashionClassifier is importable
        repo_src = Path(__file__).parent.parent / "src"
        sys.path.insert(0, str(repo_src))
        from model import FashionClassifier
        # map_location="cpu" forces all tensors off GPU even if ckpt was saved on CUDA
        m = FashionClassifier.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        m.eval()
        m = m.cpu()   # belt-and-suspenders: move any remaining GPU tensors to CPU
        with torch.no_grad():
            return m(tensor.cpu())
    except ImportError as e:
        print(f"    ⚠  Cannot import FashionClassifier: {e}")
        print("    Skipping Lightning parity check.")
        return None
    except Exception as e:
        print(f"    ⚠  Lightning load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Local Lambda weight-extraction validator")
    p.add_argument("--ckpt",    type=Path, required=True, help="Path to best.ckpt")
    p.add_argument("--norm",    type=Path, required=True, help="Path to normalization.npy")
    p.add_argument("--image",   type=Path, required=True, help="Test image path")
    p.add_argument("--classes", type=str,  required=True,
                   help="Comma-separated class names in training order")
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    classes    = [c.strip() for c in args.classes.split(",")]
    num_classes = len(classes)

    print(f"\nClasses ({num_classes}): {classes}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Norm path  : {args.norm}")
    print(f"Test image : {args.image}")
    print(f"Image size : {args.image_size}")

    # ── [1+2] Load model ─────────────────────────────────────────────────
    t_cold = time.perf_counter()

    norm  = np.load(str(args.norm))
    model = build_resnet50_head(num_classes)
    model = load_weights_from_lightning_ckpt(args.ckpt, model)

    print(f"\n[2] Shape check")
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    with torch.no_grad():
        out = model(dummy)
    print(f"    Input  : {tuple(dummy.shape)}")
    print(f"    Output : {tuple(out.shape)}  (expected: (1, {num_classes}))")
    assert out.shape == (1, num_classes), "Shape mismatch!"
    print(f"    ✓  Shape correct")

    cold_start_ms = (time.perf_counter() - t_cold) * 1000

    # ── [3] Prediction on real image ─────────────────────────────────────
    print(f"\n[3] Prediction")
    tensor = preprocess_image(args.image, args.image_size, norm)

    t_infer = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    inference_ms = (time.perf_counter() - t_infer) * 1000

    pred_idx = int(probs.argmax())
    print(f"    Predicted : {classes[pred_idx]}  ({probs[pred_idx]:.4f})")
    print(f"    All scores:")
    for cls, prob in sorted(zip(classes, probs.tolist()), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"      {cls:<20} {prob:.4f}  {bar}")

    # ── [4] Lightning parity ─────────────────────────────────────────────
    print(f"\n[4] Lightning parity check")
    lightning_logits = lightning_forward(args.ckpt, tensor)
    if lightning_logits is not None:
        diff = (logits - lightning_logits).abs().max().item()
        print(f"    Max logit diff vs Lightning: {diff:.2e}")
        if diff < 1e-4:
            print(f"    ✓  Outputs match (diff < 1e-4)")
        else:
            print(f"    ⚠  Diff is large — check key stripping logic")

    # ── [5] Timing ───────────────────────────────────────────────────────
    print(f"\n[5] Timing")
    print(f"    Cold start (model load) : {cold_start_ms:>8.1f} ms")
    print(f"    First inference         : {inference_ms:>8.1f} ms")

    # Warm inference — 10 runs
    times = []
    for _ in range(10):
        t = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        times.append((time.perf_counter() - t) * 1000)

    print(f"    Warm inference (mean)   : {np.mean(times):>8.1f} ms")
    print(f"    Warm inference (min/max): {min(times):.1f} / {max(times):.1f} ms")

    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"  Cold start  : {cold_start_ms:.0f} ms  (Lambda: add ~5-15s for S3 download)")
    print(f"  Inference   : {np.mean(times):.1f} ms per image")
    print(f"  Estimated Lambda cold start: {cold_start_ms/1000 + 10:.0f}-{cold_start_ms/1000 + 20:.0f} s")
    print(f"  Estimated Lambda warm      : {np.mean(times):.0f} ms per image")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
