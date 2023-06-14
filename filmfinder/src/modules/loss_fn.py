import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BalancedLogLoss(nn.Module):
    def __init__(self, class_weight, device="cuda"):
        super(BalancedLogLoss, self).__init__()
        self.class_weight = torch.tensor(
            class_weight, dtype=torch.float32, device=device
        )

    def forward(self, y_pred, label):
        y_pred = torch.sigmoid(y_pred)

        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        log_loss = -(label * torch.log(y_pred) + (1 - label) * torch.log(1 - y_pred))

        log_loss = torch.mean(log_loss, dim=0)
        weighted_log_loss = self.class_weight * log_loss
        balanced_log_loss = torch.sum(weighted_log_loss) / torch.sum(self.class_weight)
        return balanced_log_loss


def balanced_log_loss(y_pred, label, class_weight):
    y_pred = 1 / (1 + np.exp(-y_pred))

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    log_loss = -(label * np.log(y_pred) + (1 - label) * np.log(1 - y_pred))

    log_loss = np.mean(log_loss, axis=0)
    weighted_log_loss = class_weight * log_loss
    balanced_log_loss = np.sum(weighted_log_loss) / np.sum(class_weight)
    return balanced_log_loss


if __name__ == "__main__":
    label = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    pred = [[0.999, 0.999, 0.999], [0.999, 0.999, 0.999], [0.999, 0.999, 0.999]]
