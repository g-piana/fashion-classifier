"""
Smoke test — verifies the full pipeline runs end-to-end in ~seconds.
Uses a tiny in-memory synthetic dataset, no real images needed.
Run with:  pytest tests/test_smoke.py -v
"""
import numpy as np
import torch
import pytest
from pathlib import Path
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classes():
    return ["square", "v_rect", "h_rect"]


@pytest.fixture
def normalization():
    # mean=128, std=50 — plausible for uint8 images scaled to float
    return np.array([[128.0, 128.0, 128.0],
                     [50.0,  50.0,  50.0]], dtype=np.float32)


@pytest.fixture
def tiny_npy_dir(tmp_path, classes):
    """
    Write 12 tiny 224x224x3 .npy images to a temp folder.
    4 per class — enough for a train/val split.
    """
    stems = []
    labels = []
    for i, cls in enumerate(classes):
        for j in range(4):
            stem = f"{cls}_{j:03d}"
            img  = np.full((224, 224, 3), i * 80, dtype=np.float32)
            np.save(tmp_path / f"{stem}.npy", img)
            stems.append(stem)
            labels.append(cls)
    return tmp_path, stems, labels


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestFashionDataset:

    def test_single_label_length(self, tiny_npy_dir, normalization, classes):
        from dataset import FashionDataset
        npy_root, stems, labels = tiny_npy_dir
        ds = FashionDataset(stems, labels, npy_root, normalization, classes,
                            label_type="single")
        assert len(ds) == 12

    def test_single_label_shapes(self, tiny_npy_dir, normalization, classes):
        from dataset import FashionDataset
        npy_root, stems, labels = tiny_npy_dir
        ds = FashionDataset(stems, labels, npy_root, normalization, classes,
                            label_type="single")
        img, lbl = ds[0]
        assert img.shape == (3, 224, 224)
        assert lbl.dtype == torch.long
        assert 0 <= lbl.item() < len(classes)

    def test_multi_label_shapes(self, tiny_npy_dir, normalization, classes):
        from dataset import FashionDataset
        npy_root, stems, labels = tiny_npy_dir
        # for multi-label test, labels are underscore-joined strings
        multi_labels = [f"{l}" for l in labels]
        ds = FashionDataset(stems, multi_labels, npy_root, normalization, classes,
                            label_type="multi")
        img, lbl = ds[0]
        assert img.shape == (3, 224, 224)
        assert lbl.shape == (len(classes),)
        assert lbl.dtype == torch.float32

    def test_normalization_applied(self, tiny_npy_dir, normalization, classes):
        from dataset import FashionDataset
        npy_root, stems, labels = tiny_npy_dir
        ds = FashionDataset(stems, labels, npy_root, normalization, classes)
        img, _ = ds[0]
        # raw pixel value for class 0 is 0.0 → normalised = (0 - 128) / 50
        expected = (0.0 - 128.0) / 50.0
        assert abs(img.mean().item() - expected) < 0.01

    def test_dataloader_batch(self, tiny_npy_dir, normalization, classes):
        from dataset import FashionDataset
        npy_root, stems, labels = tiny_npy_dir
        ds = FashionDataset(stems, labels, npy_root, normalization, classes)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        imgs, lbls = next(iter(loader))
        assert imgs.shape == (4, 3, 224, 224)
        assert lbls.shape == (4,)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestFashionClassifier:

    def test_forward_resnet18(self, classes):
        from model import FashionClassifier
        model = FashionClassifier(
            num_classes=len(classes),
            backbone="resnet18",
            pretrained=False,    # no download in tests
            label_type="single",
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, len(classes))

    def test_forward_resnet50(self, classes):
        from model import FashionClassifier
        model = FashionClassifier(
            num_classes=len(classes),
            backbone="resnet50",
            pretrained=False,
            label_type="single",
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, len(classes))

    def test_freeze_backbone(self, classes):
        from model import FashionClassifier
        model = FashionClassifier(
            num_classes=len(classes),
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=True,
            label_type="single",
        )
        frozen = [n for n, p in model.named_parameters()
                  if not p.requires_grad]
        trainable = [n for n, p in model.named_parameters()
                     if p.requires_grad]
        assert len(frozen) > 0,    "Expected frozen params"
        assert len(trainable) > 0, "Expected trainable params (fc head)"
        assert all("fc" in n for n in trainable)


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_one_epoch(self, tiny_npy_dir, normalization, classes):
        """
        Full train loop for 1 epoch on 12 tiny images.
        Verifies dataset → dataloader → model → lightning trainer
        all wire together without errors.
        """
        import lightning as L
        from dataset import FashionDataset
        from model import FashionClassifier

        npy_root, stems, labels = tiny_npy_dir

        train_ds = FashionDataset(stems[:8], labels[:8], npy_root,
                                  normalization, classes, label_type="single")
        val_ds   = FashionDataset(stems[8:], labels[8:], npy_root,
                                  normalization, classes, label_type="single")

        train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=4, num_workers=0)

        model = FashionClassifier(
            num_classes=len(classes),
            backbone="resnet18",
            pretrained=False,
            label_type="single",
            learning_rate=1e-3,
        )

        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,          # no mlflow in tests
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        # Should complete without raising
        trainer.fit(model, train_loader, val_loader)