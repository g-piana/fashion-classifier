"""
evaluate.py
===========
Loads a trained checkpoint and evaluates it on a CSV-defined dataset.
Produces:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Confusion matrix (console + saved as PNG)
  - Per-class accuracy breakdown
  - Optional: saves misclassified image stems to a CSV for inspection

Usage
-----
    python src/evaluate.py \
        category=jackets \
        data=jackets \
        filesystem=deepfashion \
        evaluate.checkpoint="E:/fashion-data/weights/jackets/01/best.ckpt" \
        evaluate.split=val          # "val", "train", or "all"
        evaluate.save_errors=true   # optional: save misclassified stems
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from dataset import FashionDataset
from model import FashionClassifier


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a normalised confusion matrix as a PNG."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


def plot_per_class_accuracy(
    cm: np.ndarray,
    classes: list[str],
    out_path: Path,
) -> None:
    """Bar chart of per-class accuracy."""
    per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 4))
    bars = ax.bar(classes, per_class_acc, color="steelblue", edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.axhline(per_class_acc.mean(), color="red", linestyle="--",
               label=f"Mean {per_class_acc.mean():.2f}")
    ax.legend()
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Per-class accuracy saved → {out_path}")


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:

    # ------------------------------------------------------------------ #
    # 1. Resolve paths and evaluate config
    # ------------------------------------------------------------------ #
    root     = Path(config.filesystem.root)
    npy_path = root / config.filesystem.npy / config.category.name / config.data.run
    csv_path = root / config.filesystem.csv / config.category.csv_file
    wts_path = root / config.filesystem.weights / config.category.name / config.data.run

    eval_cfg   = config.get("evaluate", {})
    ckpt_path  = eval_cfg.get("checkpoint", str(wts_path / "best.ckpt"))
    split      = eval_cfg.get("split", "val")        # "val", "train", "all"
    save_errors = eval_cfg.get("save_errors", False)

    label_col  = config.category.label_column
    classes    = list(config.category.classes)
    label_type = config.category.label_type

    print(f"\n{'='*55}")
    print(f"Category   : {config.category.name}")
    print(f"Classes    : {classes}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Split      : {split}")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------ #
    # 2. Load labels and build the requested split
    # ------------------------------------------------------------------ #
    df = pd.read_csv(csv_path)
    df = df[df[label_col].isin(classes)].reset_index(drop=True)

    seed      = config.training.seed
    val_split = config.training.val_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df[label_col],
        random_state=seed,           # MUST match training seed for valid val split
    )

    if split == "val":
        eval_df = val_df.reset_index(drop=True)
    elif split == "train":
        eval_df = train_df.reset_index(drop=True)
    else:  # "all"
        eval_df = df.reset_index(drop=True)

    print(f"Evaluating on {len(eval_df)} images ({split} split)\n")

    # ------------------------------------------------------------------ #
    # 3. Load normalization — always use the saved training stats
    # ------------------------------------------------------------------ #
    norm_path = wts_path / "normalization.npy"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {norm_path}\n"
            "Run train.py first to generate it."
        )
    normalization = np.load(norm_path)
    print(f"Loaded normalization from {norm_path}")

    # ------------------------------------------------------------------ #
    # 4. Dataset and dataloader — no augmentation at eval time
    # ------------------------------------------------------------------ #
    dataset = FashionDataset(
        image_stems=eval_df["name"].tolist(),
        labels_raw=eval_df[label_col].tolist(),
        npy_root=npy_path,
        normalization=normalization,
        classes=classes,
        label_type=label_type,
        augment=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------ #
    # 5. Load model from checkpoint
    # ------------------------------------------------------------------ #
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = FashionClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded from checkpoint  (device: {device})\n")

    # ------------------------------------------------------------------ #
    # 6. Run inference
    # ------------------------------------------------------------------ #
    all_preds  = []
    all_labels = []
    all_stems  = eval_df["name"].tolist()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images  = images.to(device)
            logits  = model(images)

            if label_type == "single":
                preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
                lbls  = targets.numpy()
            else:
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                lbls  = targets.numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * loader.batch_size} / {len(dataset)}")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ------------------------------------------------------------------ #
    # 7. Metrics
    # ------------------------------------------------------------------ #
    print(f"\n{'='*55}")
    print(f"RESULTS — {config.category.name} ({split} split)")
    print(f"{'='*55}\n")

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy : {overall_acc:.4f}  ({overall_acc*100:.1f}%)\n")

    print("Per-class report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=classes,
        digits=3,
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())
    print()

    # Per-class accuracy
    per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    print("Per-class accuracy:")
    for cls, acc in zip(classes, per_class_acc):
        bar = "█" * int(acc * 20)
        print(f"  {cls:15s} {acc:.3f}  {bar}")

    # ------------------------------------------------------------------ #
    # 8. Save plots
    # ------------------------------------------------------------------ #
    out_dir = wts_path / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm, classes,
        out_path=out_dir / f"confusion_matrix_{split}.png",
        title=f"{config.category.name} — {split} split  (acc={overall_acc:.3f})",
    )
    plot_per_class_accuracy(
        cm, classes,
        out_path=out_dir / f"per_class_accuracy_{split}.png",
    )

    # Save metrics CSV
    metrics_df = pd.DataFrame({
        "class":    classes,
        "accuracy": per_class_acc,
        "n_samples": cm.sum(axis=1),
    })
    metrics_df.loc[len(metrics_df)] = ["OVERALL", overall_acc, len(eval_df)]
    metrics_df.to_csv(out_dir / f"metrics_{split}.csv", index=False)
    print(f"  Metrics CSV saved → {out_dir / f'metrics_{split}.csv'}")

    # ------------------------------------------------------------------ #
    # 9. Save misclassified stems (optional)
    # ------------------------------------------------------------------ #
    if save_errors and label_type == "single":
        errors = [
            {
                "name":       all_stems[i],
                "true_label": classes[all_labels[i]],
                "pred_label": classes[all_preds[i]],
            }
            for i in range(len(all_preds))
            if all_preds[i] != all_labels[i]
        ]
        errors_df = pd.DataFrame(errors)
        errors_path = out_dir / f"errors_{split}.csv"
        errors_df.to_csv(errors_path, index=False)
        print(f"\n  Misclassified : {len(errors)} / {len(eval_df)} images")
        print(f"  Errors CSV    → {errors_path}")

        # Most common confusions
        print("\n  Top confusions (true → predicted):")
        confusion_counts = (
            errors_df.groupby(["true_label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        print(confusion_counts.to_string(index=False))

    print(f"\nAll evaluation outputs → {out_dir}\n")


if __name__ == "__main__":
    main()
