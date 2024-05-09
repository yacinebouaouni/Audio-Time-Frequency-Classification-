import pytorch_lightning as pl
from .model import EffNet
from .loss import get_loss
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
from ..metric import score
import numpy as np
import pandas as pd


class BirdModel(pl.LightningModule):
    def __init__(self, config, label_list):
        super().__init__()

        # == config ==
        self.config = config
        # == backbone ==
        self.backbone = EffNet(config.MODEL_TYPE, n_classes=len(label_list))

        # == loss function ==
        self.loss_fn = get_loss(config)

        # == record ==
        self.validation_step_outputs = []

        # == classes ==
        self.label_list = label_list

    def forward(self, images):
        return self.backbone(images)

    def configure_optimizers(self):
        # == define optimizer ==
        model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # == define learning rate scheduler ==
        lr_scheduler = CosineAnnealingWarmRestarts(
            model_optimizer,
            T_0=self.config.EPOCHS,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1,
        )

        return {
            "optimizer": model_optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        # == obtain input and target ==
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        # == pred ==
        y_pred = self(image)

        # == compute loss ==
        train_loss = self.loss_fn(y_pred, target)  # type: ignore

        # == record ==
        self.log("train_loss", train_loss, True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # == obtain input and target ==
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        # == pred ==
        with torch.no_grad():
            y_pred = self(image)

        self.validation_step_outputs.append({"logits": y_pred, "targets": target})

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def on_validation_epoch_end(self):
        # = merge batch data =
        outputs = self.validation_step_outputs

        output_val = (
            nn.Softmax(dim=1)(torch.cat([x["logits"] for x in outputs], dim=0))
            .cpu()
            .detach()
        )
        target_val = torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach()

        # = compute validation loss =
        val_loss = self.loss_fn(output_val, target_val)  # type: ignore

        # target to one-hot
        target_val = torch.nn.functional.one_hot(target_val, len(self.label_list))

        # = val with ROC AUC =
        gt_df = pd.DataFrame(
            target_val.numpy().astype(np.float32), columns=self.label_list
        )
        pred_df = pd.DataFrame(
            output_val.numpy().astype(np.float32), columns=self.label_list
        )

        gt_df["id"] = [f"id_{i}" for i in range(len(gt_df))]
        pred_df["id"] = [f"id_{i}" for i in range(len(pred_df))]

        val_score = score(gt_df, pred_df, row_id_column_name="id")

        self.log("val_score", val_score, True)

        # clear validation outputs
        self.validation_step_outputs = list()

        return {"val_loss": val_loss, "val_score": val_score}
