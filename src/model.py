import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import lightning as L
from torchmetrics import Accuracy, F1Score
from transformers import CLIPModel

BACKBONE_FC_FEATURES = {"resnet18": 512, "resnet50": 2048, "clip_vit": 512}

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class CLIPImageEncoder(nn.Module):
    """
    Vision-only wrapper around FashionCLIP.
    Outputs the projected 512-d embedding. Text encoder is never loaded.
    freeze=True  → Stage 1: head only.
    freeze=False + unfreeze_last_n=N → Stage 2: last N transformer blocks.
    """
    def __init__(self, model_id="patrickjohncyh/fashion-clip",
                 freeze=True, unfreeze_last_n=0):
        super().__init__()
        clip = CLIPModel.from_pretrained(model_id)
        self.vision_model      = clip.vision_model
        self.visual_projection = clip.visual_projection

        # Freeze entire backbone by default
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N encoder blocks (Stage 2)
        if not freeze or unfreeze_last_n > 0:
            n = unfreeze_last_n if freeze else len(self.vision_model.encoder.layers)
            for layer in self.vision_model.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.visual_projection.parameters():
                p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # out = self.vision_model(pixel_values=pixel_values)
        out = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        return self.visual_projection(out.pooler_output)   # (B, 512)


class FashionClassifier(L.LightningModule):
    """
    Unified model: ResNet18 | ResNet50 | CLIPViT backbone.
    Single-label and multi-label classification.
    All behaviour driven by Hydra config — no code changes needed to switch.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        label_type: str = "single",
        learning_rate: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        clip_model_id: str = "patrickjohncyh/fashion-clip",
        clip_unfreeze_layers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Backbone ──────────────────────────────────────────────────────
        if backbone == "clip_vit":
            self.encoder = CLIPImageEncoder(
                model_id=clip_model_id,
                freeze=freeze_backbone,
                unfreeze_last_n=clip_unfreeze_layers,
            )
            # Two-layer head for better generalisation on small datasets
            self.head = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
            self._is_clip = True

        elif backbone in ("resnet18", "resnet50"):
            if backbone == "resnet50":
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                self.model = torchvision.models.resnet50(weights=weights)
            else:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                self.model = torchvision.models.resnet18(weights=weights)
            in_features = BACKBONE_FC_FEATURES[backbone]
            self.model.fc = nn.Linear(in_features, num_classes)
            if freeze_backbone:
                for name, param in self.model.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
            self._is_clip = False

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Loss and metrics ──────────────────────────────────────────────
        if label_type == "single":
            self.loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_f1    = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        else:
            self.loss_fn   = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            self.train_acc = Accuracy(task="multilabel", num_labels=num_classes)
            self.val_acc   = Accuracy(task="multilabel", num_labels=num_classes)
            self.val_f1    = F1Score(task="multilabel", num_labels=num_classes, average="macro")

        self.label_type    = label_type
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_clip:
            return self.head(self.encoder(x))
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss   = self.loss_fn(logits, targets)
        self.train_acc.update(self._to_preds(logits), targets)
        self.log("train_loss", loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss   = self.loss_fn(logits, targets)
        preds  = self._to_preds(logits)
        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.log("val_loss", loss,          on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",  self.val_acc,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1",   self.val_f1,   on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def _to_preds(self, logits: torch.Tensor) -> torch.Tensor:
        if self.label_type == "single":
            return torch.softmax(logits, dim=1).argmax(dim=1)
        return (torch.sigmoid(logits) > 0.5).int()
