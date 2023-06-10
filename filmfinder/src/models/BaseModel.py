import torch.nn as nn
from transformers import BertModel


class BaseModel(nn.Module):
    def __init__(self, pretrain_model, num_classes, freeze_bert=False):
        super(BaseModel, self).__init__()
        self.model = BertModel.from_pretrained(pretrain_model)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
        classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.model.classifier = classifier
