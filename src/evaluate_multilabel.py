"""
evaluate_multilabel.py
======================
Evaluates a trained multi-label checkpoint.

Produces:
  - Per-attribute F1, precision, recall, support
  - Threshold sweep per attribute (finds optimal threshold)
  - Prediction positive-rate vs ground-truth positive-rate (catches bias)
  - Per-attribute bar chart saved as PNG

Usage
-----
    python src/evaluate_multilabel.py \
        category=model_details \
        model=clip_vit \
        data=model_details \
        training=multilabel \
        filesystem=local

    # Evaluate on full dataset instead of val split:
    python src/evaluate_multilabel.py ... evaluate.split=all

    # Use a specific checkpoint:
    python src/evaluate_multilabel.py ... evaluate.checkpoint="E:/path/to/best.ckpt"
"""

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import FashionDataset
from model import FashionClassifier


def threshold_sweep(
    scores: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    thresholds: list[float] = None,
) -> pd.DataFrame:
    """Find the F1-maximising threshold for each attribute independently."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]

    rows = []
    for ci, cls in enumerate(classes):
        best_f1, best_thresh = 0.0, 0.5
        for t in thresholds:
            preds = (scores[:, ci] >= t).astype(int)
            f1 = f1_score(targets[:, ci], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        rows.append({"attribute": cls, "best_threshold": best_thresh, "best_f1": round(best_f1, 3)})
    return pd.DataFrame(rows)


def plot_per_attribute(df_metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    classes = df_metrics["attribute"].tolist()
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 1.4), 5))
    ax.bar(x - width, df_metrics["precision"], width, label="Precision", color="#4e79a7")
    ax.bar(x,         df_metrics["recall"],    width, label="Recall",    color="#f28e2b")
    ax.bar(x + width, df_metrics["f1"],        width, label="F1",        color="#59a14f")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.4)

    for xi, f1 in zip(x, df_metrics["f1"]):
        ax.text(xi + width, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:

    root     = Path(config.filesystem.root)
    npy_path = root / config.filesystem.npy / config.category.name / config.data.run
    csv_path = root / config.filesystem.csv / config.category.csv_file
    wts_path = root / config.filesystem.weights / config.category.name / config.data.run

    eval_cfg  = config.get("evaluate", {})
    ckpt_path = eval_cfg.get("checkpoint", "").strip() or str(wts_path / "best.ckpt")
    split     = eval_cfg.get("split", "val")

    classes   = list(config.category.classes)
    label_col = config.category.label_column

    print(f"\n{'='*55}")
    print(f"Category   : {config.category.name}")
    print(f"Classes    : {classes}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Split      : {split}")
    print(f"{'='*55}\n")

    # Load CSV — keep all rows, treat missing attributes as clean negatives
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["name"]).reset_index(drop=True)
    df[label_col] = df[label_col].fillna("")

    train_df, val_df = train_test_split(
        df, test_size=config.training.val_split, random_state=config.training.seed
    )

    if split == "val":
        eval_df = val_df.reset_index(drop=True)
    elif split == "train":
        eval_df = train_df.reset_index(drop=True)
    else:
        eval_df = df.reset_index(drop=True)

    print(f"Evaluating on {len(eval_df)} images ({split} split)\n")

    norm_path = wts_path / "normalization.npy"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization not found: {norm_path}")
    normalization = np.load(norm_path)

    dataset = FashionDataset(
        image_stems=eval_df["name"].tolist(),
        labels_raw=eval_df[label_col].tolist(),
        npy_root=npy_path,
        normalization=normalization,
        classes=classes,
        label_type="multi",
        augment=None,
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=4,
                        shuffle=False, persistent_workers=True)

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = FashionClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded  (device: {device})\n")

    all_scores, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.numpy())

    all_scores  = np.vstack(all_scores)   # (N, C)
    all_targets = np.vstack(all_targets)  # (N, C)

    # --- Metrics at threshold=0.5 ---
    preds_05 = (all_scores >= 0.5).astype(int)

    print(f"{'='*55}")
    print(f"RESULTS at threshold=0.50  ({split} split)")
    print(f"{'='*55}\n")

    metric_rows = []
    for ci, cls in enumerate(classes):
        gt, pred = all_targets[:, ci], preds_05[:, ci]
        metric_rows.append({
            "attribute":      cls,
            "precision":      round(precision_score(gt, pred, zero_division=0), 3),
            "recall":         round(recall_score(gt, pred, zero_division=0), 3),
            "f1":             round(f1_score(gt, pred, zero_division=0), 3),
            "support":        int(gt.sum()),
            "pred_positives": int(pred.sum()),
        })

    df_metrics = pd.DataFrame(metric_rows)

    header = f"  {'Attribute':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8} {'Pred+':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in df_metrics.iterrows():
        flag = "  ✓" if row["f1"] >= 0.80 else ("  ~" if row["f1"] >= 0.65 else "  ✗")
        print(f"  {row['attribute']:<18} {row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{row['f1']:>6.3f} {row['support']:>8} {row['pred_positives']:>6}{flag}")

    macro_f1 = df_metrics["f1"].mean()
    print(f"\n  Macro F1 (mean): {macro_f1:.3f}")
    print(f"\n  Legend: ✓ >=0.80 ships  ~ 0.65-0.79 fine-tune  ✗ <0.65 needs more data\n")

    # --- Threshold sweep ---
    print(f"{'='*55}")
    print(f"THRESHOLD SWEEP (per-attribute optimum)")
    print(f"{'='*55}\n")
    df_thresh = threshold_sweep(all_scores, all_targets, classes)
    print(df_thresh.to_string(index=False))
    print()

    # --- Save outputs ---
    out_dir = wts_path / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics.to_csv(out_dir / f"multilabel_metrics_{split}.csv", index=False)
    df_thresh.to_csv(out_dir / f"multilabel_thresholds_{split}.csv", index=False)
    plot_per_attribute(
        df_metrics,
        out_path=out_dir / f"multilabel_per_attribute_{split}.png",
        title=f"{config.category.name} — {split} split  (macro F1={macro_f1:.3f})",
    )

    print(f"All outputs -> {out_dir}\n")


if __name__ == "__main__":
    main()
