import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy


class BertMultiLabelClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5):
        super(BertMultiLabelClassifier, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels.float())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels.float())
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
