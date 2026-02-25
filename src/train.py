import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import albumentations as A
from pathlib import Path

from dataset import FashionDataset
from model import FashionClassifier


def compute_normalization(stems: list[str], npy_root: Path) -> np.ndarray:
    """
    Compute per-channel mean and std over the training set.
    Returns array of shape (2, 3): [mean_per_channel, std_per_channel]
    """
    print(f"  Computing normalization over {len(stems)} training images...")
    total       = np.zeros(3, dtype=np.float64)
    total_sq    = np.zeros(3, dtype=np.float64)
    count       = 0

    for stem in stems:
        img = np.load(npy_root / f"{stem}.npy").astype(np.float64)  # HxWx3
        total    += img.mean(axis=(0, 1))
        total_sq += (img ** 2).mean(axis=(0, 1))
        count    += 1

    mean = total / count
    std  = np.sqrt(total_sq / count - mean ** 2)
    std  = np.maximum(std, 1e-8)   # avoid division by zero

    print(f"  mean={mean.round(2)}  std={std.round(2)}")
    return np.array([mean, std], dtype=np.float32)


def compute_class_weights(
    df: pd.DataFrame,
    label_col: str,
    classes: list[str],
) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle class imbalance.
    Returns a float tensor of shape (num_classes,).
    """
    counts = df[label_col].value_counts().to_dict()
    freqs  = np.array([counts.get(c, 0) for c in classes], dtype=np.float32)
    # avoid div by zero for classes not present in training split
    freqs  = np.where(freqs == 0, 1, freqs)
    weights = 1.0 / freqs
    weights = weights / weights.sum()   # normalise to sum=1
    return torch.tensor(weights, dtype=torch.float32)


def build_augmentation(aug_cfg) -> A.Compose:
    ops = []
    if aug_cfg.get("gamma_contrast", False):
        ops.append(A.RandomGamma(p=0.5))
    scale = aug_cfg.get("scale", None)
    translate = aug_cfg.get("translate_px", None)
    rotate = aug_cfg.get("rotate", None)
    if scale or translate or rotate:
        ops.append(A.Affine(
            scale=tuple(scale) if scale else None,
            translate_px={"x": tuple(translate), "y": tuple(translate)} if translate else None,
            rotate=tuple(rotate) if rotate else None,
            p=0.7,
        ))
    return A.Compose(ops)


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig) -> None:

    # ------------------------------------------------------------------ #
    # 1. Resolve paths
    # ------------------------------------------------------------------ #
    root     = Path(config.filesystem.root)
    npy_path = root / config.filesystem.npy / config.category.name / config.data.run
    csv_path = root / config.filesystem.csv / config.category.csv_file
    wts_path = root / config.filesystem.weights / config.category.name / config.data.run
    wts_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2. Load and filter labels
    # ------------------------------------------------------------------ #
    label_col = config.category.label_column
    classes   = list(config.category.classes)
    label_type = config.category.label_type

    df = pd.read_csv(csv_path)
    df = df[df[label_col].isin(classes)].reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"Category   : {config.category.name}")
    print(f"Run        : {config.data.run}")
    print(f"Classes    : {classes}")
    print(f"Label type : {label_type}")
    print(f"Samples    : {len(df)}")
    print(f"{'='*55}\n")

    if len(df) == 0:
        raise ValueError("No valid rows found. Check category config and CSV.")

    # ------------------------------------------------------------------ #
    # 3. Train / val split  (reproducible via seed)
    # ------------------------------------------------------------------ #
    seed      = config.training.seed
    val_split = config.training.val_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df[label_col],   # preserve class balance in both splits
        random_state=seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Seed          : {seed}\n")

    # ------------------------------------------------------------------ #
    # 4. Normalization  (computed on train only, saved with checkpoint)
    # ------------------------------------------------------------------ #
    norm_path = wts_path / "normalization.npy"
    if norm_path.exists():
        print(f"Loading existing normalization from {norm_path}")
        normalization = np.load(norm_path)
    else:
        normalization = compute_normalization(
            stems=train_df["name"].tolist(),
            npy_root=npy_path,
        )
        np.save(norm_path, normalization)
        print(f"Normalization saved → {norm_path}\n")

    # ------------------------------------------------------------------ #
    # 5. Datasets and dataloaders
    # ------------------------------------------------------------------ #
    augmentation = build_augmentation(config.training.augmentation)

    train_dataset = FashionDataset(
        image_stems=train_df["name"].tolist(),
        labels_raw=train_df[label_col].tolist(),
        npy_root=npy_path,
        normalization=normalization,
        classes=classes,
        label_type=label_type,
        augment=augmentation,
    )
    val_dataset = FashionDataset(
        image_stems=val_df["name"].tolist(),
        labels_raw=val_df[label_col].tolist(),
        npy_root=npy_path,
        normalization=normalization,   # same train norm — intentional
        classes=classes,
        label_type=label_type,
        augment=None,                  # no augmentation at val time
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------ #
    # 6. Class weights
    # ------------------------------------------------------------------ #
    class_weights = None
    if config.training.use_class_weights:
        class_weights = compute_class_weights(train_df, label_col, classes)
        print(f"Class weights: {dict(zip(classes, class_weights.numpy().round(3)))}\n")

    # ------------------------------------------------------------------ #
    # 7. Model
    # ------------------------------------------------------------------ #
    # finetune.yaml sets freeze_backbone: true — read from training config
    # if present, otherwise fall back to model config
    freeze = config.training.get("freeze_backbone", config.model.freeze_backbone)

    model = FashionClassifier(
        num_classes=len(classes),
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        freeze_backbone=freeze,
        label_type=label_type,
        learning_rate=config.training.learning_rate,
        class_weights=class_weights,
    )

    # ------------------------------------------------------------------ #
    # 8. Callbacks and logger
    # ------------------------------------------------------------------ #
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(wts_path),
        filename="best",
        monitor=config.training.checkpoint_metric,
        mode=config.training.checkpoint_mode,
        save_top_k=1,
        save_weights_only=False,   # full checkpoint so hparams travel with it
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=config.category.name,
        run_name=f"{config.model.backbone}_{config.data.run}",
        tracking_uri=config.filesystem.mlflow_uri,
    )

    # Log the full resolved Hydra config as MLflow params
    from omegaconf import OmegaConf
    flat_cfg = OmegaConf.to_container(config, resolve=True)
    mlflow_logger.log_hyperparams({"config": str(flat_cfg)})

    # ------------------------------------------------------------------ #
    # 9. Trainer
    # ------------------------------------------------------------------ #
    torch.set_float32_matmul_precision("medium")

    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",        # uses GPU if available, CPU otherwise
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
    )

    # ------------------------------------------------------------------ #
    # 10. Train
    # ------------------------------------------------------------------ #
    print(f"Starting training — {config.training.max_epochs} epochs")
    print(f"Checkpoint → {wts_path}/best.ckpt\n")

    trainer.fit(model, train_loader, val_loader)

    print(f"\nBest checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Best {config.training.checkpoint_metric} : "
          f"{checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()