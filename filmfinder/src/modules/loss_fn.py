import numpy as np
import torch
import torch.nn.functional as F


class BalancedLogLoss(torch.nn.Module):
    def __init__(self, num_classes, eps=1e-15):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, input, target):
        num_obs = target.sum(dim=0)
        class_weight = num_obs.sum() / (self.num_classes * num_obs)

        log_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        weighted_log_loss = class_weight * target * log_loss
        balanced_log_loss = weighted_log_loss.sum() / num_obs.sum()
        return balanced_log_loss


def balanced_log_loss(y_pred, label):
    num_classes = label.shape[1]
    num_obs = np.sum(label, axis=0)
    class_weight = num_obs.sum() / (num_classes * num_obs)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    log_loss = -(label * np.log(y_pred) + (1 - label) * np.log(1 - y_pred))
    weighted_log_loss = class_weight * label * log_loss
    balanced_log_loss = np.sum(weighted_log_loss) / num_obs.sum()
    return balanced_log_loss
