import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import numpy as np
import cv2


# Supported extensions in priority order
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def find_image(folder: Path, stem: str) -> Path | None:
    """Find an image file by stem, trying multiple extensions."""
    for ext in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad image to square with white background, content centered."""
    h, w = img.shape[:2]
    if w == h:
        return img
    size = max(w, h)
    result = np.full((size, size, 3), 255, dtype=img.dtype)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    result[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return result


def load_and_prepare(image_path: Path, target_size: int) -> np.ndarray | None:
    """Load image, pad to square, resize. Returns HxWx3 float32 or None on failure."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pad_to_square(img)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)
    except Exception as e:
        print(f"  WARNING: failed to process {image_path.name}: {e}")
        return None


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:

    # --- Resolve paths ---
    root     = Path(config.filesystem.root)
    raw_sub  = config.data.get("raw_subdir", "")
    raw_path = root / config.filesystem.raw / raw_sub if raw_sub else root / config.filesystem.raw
    npy_path = root / config.filesystem.npy / config.category.name / config.data.run
    csv_path = root / config.filesystem.csv / config.category.csv_file

    npy_path.mkdir(parents=True, exist_ok=True)

    target_size: int = config.data.image_size
    max_frames: int  = config.data.max_frames
    print(f"  raw_path resolved : {raw_path}")
    print(f"  raw_path exists   : {raw_path.exists()}")
    print(f"\n{'='*50}")
    print(f"Category  : {config.category.name}")
    print(f"Run       : {config.data.run}")
    print(f"CSV       : {csv_path}")
    print(f"Raw images: {raw_path}")
    print(f"Output    : {npy_path}")
    print(f"Image size: {target_size}x{target_size}")
    print(f"{'='*50}\n")

    # --- Load and filter labels ---
    labels = pd.read_csv(csv_path)
    label_col = config.category.label_column
    valid_classes = list(config.category.classes)

    before = len(labels)
    labels = labels[labels[label_col].isin(valid_classes)].reset_index(drop=True)
    print(f"CSV rows: {before} total → {len(labels)} with valid labels")
    print(f"Class distribution:\n{labels[label_col].value_counts().to_string()}\n")

    if len(labels) == 0:
        print("ERROR: no valid rows found. Check category.label_column and category.classes.")
        return

    if len(labels) > max_frames:
        labels = labels.sample(n=max_frames, random_state=0).reset_index(drop=True)
        print(f"Capped to {max_frames} frames")

    # --- Process images ---
    processed, skipped, already_done = 0, 0, 0

    for _, row in labels.iterrows():
        stem     = row["name"]
        out_file = npy_path / f"{stem}.npy"

        if out_file.exists():
            already_done += 1
            continue

        img_path = find_image(raw_path, stem)
        if img_path is None:
            print(f"  MISSING: {stem}  (tried {[stem+e for e in IMAGE_EXTENSIONS]})")
            skipped += 1
            continue

        img = load_and_prepare(img_path, target_size)
        if img is None:
            skipped += 1
            continue

        np.save(out_file, img)
        processed += 1

        if processed % 500 == 0:
            print(f"  {processed} new images saved...")

    total = processed + already_done
    print(f"\nDone.")
    print(f"  Newly processed : {processed}")
    print(f"  Already on disk : {already_done}")
    print(f"  Skipped/failed  : {skipped}")
    print(f"  Total available : {total}")
    print(f"  Output          : {npy_path}")


if __name__ == "__main__":
    main()