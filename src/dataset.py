import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A


def encode_single_label(label: str, classes: list) -> torch.Tensor:
    """Return a scalar long tensor — index of label in classes list."""
    return torch.tensor(classes.index(label), dtype=torch.long)


def encode_multi_label(labels: list[str], classes: list) -> torch.Tensor:
    """Return a float binary vector of length len(classes)."""
    vec = torch.zeros(len(classes), dtype=torch.float32)
    for lbl in labels:
        if lbl in classes:
            vec[classes.index(lbl)] = 1.0
    return vec


class FashionDataset(Dataset):
    """
    Unified dataset for both single-label (multiclass) and
    multi-label classification.

    Args:
        image_stems  : list of filename stems (no extension, no path)
        labels_raw   : list of raw label strings from the CSV
        npy_root     : folder containing the .npy image files
        normalization: np.ndarray of shape (2, 3) — [mean, std] per channel
        classes      : ordered list of class names (from config)
        label_type   : "single" or "multi"
        augment      : albumentations Compose pipeline or None
    """

    def __init__(
        self,
        image_stems: list[str],
        labels_raw: list[str],
        npy_root: Path,
        normalization: np.ndarray,
        classes: list[str],
        label_type: str = "single",
        augment: A.Compose | None = None,
    ):
        assert len(image_stems) == len(labels_raw)
        assert label_type in ("single", "multi")

        self.stems     = image_stems
        self.labels_raw = labels_raw
        self.npy_root  = Path(npy_root)
        self.mean      = normalization[0]   # shape (3,)
        self.std       = normalization[1]   # shape (3,)
        self.classes   = classes
        self.label_type = label_type
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img  = np.load(self.npy_root / f"{stem}.npy").astype(np.float32)

        if self.augment is not None:
            # albumentations expects uint8 HxWxC for most transforms
            # we cast, augment, then cast back to float32
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            img = self.augment(image=img_uint8)["image"].astype(np.float32)

        # Normalize per channel
        img = (img - self.mean) / (self.std + 1e-8)

        # HxWxC → CxHxW
        img = torch.from_numpy(img).permute(2, 0, 1)

        raw_label = self.labels_raw[idx]
        if self.label_type == "single":
            label = encode_single_label(raw_label, self.classes)
        else:
            label = encode_multi_label(raw_label.split("_"), self.classes)

        return img, label