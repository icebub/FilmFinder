import torch.nn as nn
from transformers import AdamW, BertForSequenceClassification, BertModel, BertTokenizer


class BaseModel(nn.Module):
    def __init__(self, pretrain_model, num_classes, freeze_bert=False):
        super(BaseModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            pretrain_model, num_labels=num_classes
        )
        if freeze_bert:
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = False
        classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.model.classifier = classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits
