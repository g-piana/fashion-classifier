import torch
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import lightning as L
from torchmetrics import Accuracy


# fc layer input features per backbone
BACKBONE_FC_FEATURES = {
    "resnet18": 512,
    "resnet50": 2048,
}


class FashionClassifier(L.LightningModule):
    """
    Unified model for single-label (multiclass) and multi-label classification.
    Backbone and training behaviour driven entirely by config.

    Args:
        num_classes   : number of output classes
        backbone      : "resnet18" or "resnet50"
        pretrained    : load ImageNet weights
        freeze_backbone: freeze all layers except the fc head
        label_type    : "single" (cross-entropy) or "multi" (BCE)
        learning_rate : optimizer lr
        class_weights : optional tensor for weighted loss
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
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Build backbone ---
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = torchvision.models.resnet50(weights=weights)
        elif backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = torchvision.models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet50'.")

        # Replace classification head
        in_features = BACKBONE_FC_FEATURES[backbone]
        self.model.fc = torch.nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        # --- Loss and metrics ---
        if label_type == "single":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
            self.train_acc = Accuracy(task="multilabel", num_labels=num_classes)
            self.val_acc   = Accuracy(task="multilabel", num_labels=num_classes)

        self.label_type = label_type
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.loss_fn(logits, targets)

        preds = self._to_preds(logits)
        self.train_acc.update(preds, targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.loss_fn(logits, targets)

        preds = self._to_preds(logits)
        self.val_acc.update(preds, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",  self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def _to_preds(self, logits: torch.Tensor) -> torch.Tensor:
        if self.label_type == "single":
            return torch.softmax(logits, dim=1).argmax(dim=1)
        else:
            return (torch.sigmoid(logits) > 0.5).int()