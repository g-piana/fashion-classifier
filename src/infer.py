"""
infer.py
========
Runs inference on preprocessed .npy images using a trained checkpoint.
Produces a CSV with predicted class and confidence score per image.

Usage
-----
    python src/infer.py \
        category=photo_or_draw \
        data=photo_or_draw \
        filesystem=local \
        infer.checkpoint="E:/fashion-data/weights/photo_or_draw/01/best.ckpt" \
        infer.npy_dir="E:/fashion-data/npy/photo_or_draw/01" \
        infer.out_csv="E:/fashion-data/csv/predictions_photo_or_draw.csv"

    # Optionally override normalization path:
        infer.norm_path="E:/fashion-data/weights/photo_or_draw/01/normalization.npy"
"""

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd

from model import FashionClassifier


class NpyInferenceDataset(Dataset):
    """
    Minimal dataset for inference — loads .npy files, normalises, returns stem + tensor.
    No labels needed.
    """

    def __init__(self, stems: list[str], npy_root: Path, normalization: np.ndarray):
        self.stems = stems
        self.npy_root = Path(npy_root)
        self.mean = normalization[0]   # (3,)
        self.std  = normalization[1]   # (3,)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img  = np.load(self.npy_root / f"{stem}.npy").astype(np.float32)
        img  = (img - self.mean) / (self.std + 1e-8)
        img  = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW
        return stem, img


def collate_fn(batch):
    stems  = [b[0] for b in batch]
    images = torch.stack([b[1] for b in batch])
    return stems, images


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:

    # ------------------------------------------------------------------ #
    # 1. Resolve config
    # ------------------------------------------------------------------ #
    infer_cfg = config.get("infer", {})

    root     = Path(config.filesystem.root)
    wts_path = root / config.filesystem.weights / config.category.name / config.data.run

    # npy_dir — explicit override or default run path
    npy_dir  = Path(infer_cfg.get("npy_dir", "")  or
                    root / config.filesystem.npy / config.category.name / config.data.run)

    # checkpoint — explicit override or default best.ckpt
    ckpt_path = Path(infer_cfg.get("checkpoint", "") or wts_path / "best.ckpt")

    # normalization — explicit override or default from weights dir
    norm_path = Path(infer_cfg.get("norm_path", "") or wts_path / "normalization.npy")

    # output CSV
    out_csv = Path(infer_cfg.get("out_csv", "") or
                   root / config.filesystem.csv / f"predictions_{config.category.name}.csv")

    classes    = list(config.category.classes)
    label_type = config.category.label_type
    batch_size = infer_cfg.get("batch_size", 64)

    print(f"\n{'='*55}")
    print(f"Category   : {config.category.name}")
    print(f"Classes    : {classes}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"NPY dir    : {npy_dir}")
    print(f"Norm path  : {norm_path}")
    print(f"Output CSV : {out_csv}")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------ #
    # 2. Validate paths
    # ------------------------------------------------------------------ #
    for p, name in [(ckpt_path, "Checkpoint"), (norm_path, "Normalization"), (npy_dir, "NPY dir")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # ------------------------------------------------------------------ #
    # 3. Collect stems
    # ------------------------------------------------------------------ #
    stems = sorted(p.stem for p in Path(npy_dir).glob("*.npy"))
    if not stems:
        raise ValueError(f"No .npy files found in {npy_dir}")
    print(f"Found {len(stems)} .npy files\n")

    # ------------------------------------------------------------------ #
    # 4. Load normalization and model
    # ------------------------------------------------------------------ #
    normalization = np.load(norm_path)

    model  = FashionClassifier.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"Model loaded  (device: {device})\n")

    # ------------------------------------------------------------------ #
    # 5. Dataset and dataloader
    # ------------------------------------------------------------------ #
    dataset = NpyInferenceDataset(stems, npy_dir, normalization)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------ #
    # 6. Inference
    # ------------------------------------------------------------------ #
    rows = []

    with torch.no_grad():
        for batch_stems, images in loader:
            images = images.to(device)
            logits = model(images)

            if label_type == "single":
                probs      = torch.softmax(logits, dim=1)          # (B, C)
                pred_idxs  = probs.argmax(dim=1).cpu().numpy()
                confidences = probs.max(dim=1).values.cpu().numpy()
                for j, (stem, pred_idx, conf) in enumerate(zip(batch_stems, pred_idxs, confidences)):
                    rows.append({
                        "name":       stem,
                        "prediction": classes[pred_idx],
                        "confidence": round(float(conf), 4),
                        **{f"prob_{cls}": round(float(probs[j, i].item()), 4)
                        for i, cls in enumerate(classes)},
                    })

            else:  # multi-label
                probs = torch.sigmoid(logits).cpu().numpy()        # (B, C)

                for stem, prob_vec in zip(batch_stems, probs):
                    row = {"name": stem}
                    for cls, p in zip(classes, prob_vec):
                        row[f"prob_{cls}"]  = round(float(p), 4)
                        row[f"pred_{cls}"]  = int(p >= 0.5)
                    rows.append(row)

    # ------------------------------------------------------------------ #
    # 7. Save CSV
    # ------------------------------------------------------------------ #
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Predictions saved → {out_csv}")
    print(f"Rows: {len(df)}\n")

    if label_type == "single":
        print("Prediction distribution:")
        print(df["prediction"].value_counts().to_string())
        print(f"\nMean confidence : {df['confidence'].mean():.4f}")
        print(f"Low confidence (<0.80) : {(df['confidence'] < 0.80).sum()} images")


if __name__ == "__main__":
    main()