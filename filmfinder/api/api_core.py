import json
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from filmfinder.src.models.BaseModel import BaseModel
from filmfinder.src.modules.BertMultiLabelClassifier import BertMultiLabelClassifier
from filmfinder.src.modules.loss_fn import balanced_log_loss


def load_model(exp_id, device):
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    exp_path = f"{abs_folder}/../experiments/{exp_id}"

    with open(f"{exp_path}/label_data.json", "r") as f:
        model_data = json.load(f)

    pretrain_model = model_data["pretrain_model"]
    reverse_mapping = model_data["reverse_mapping"]
    num_class = model_data["num_class"]

    with open(f"{exp_path}/eval_data.json", "r") as f:
        eval_data = json.load(f)

    thresholds = eval_data["thresholds"]

    model_checkpoint = f"{exp_path}/best_model.ckpt"

    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    model = BaseModel(pretrain_model, num_classes=num_class, freeze_bert=True)

    pl_module = BertMultiLabelClassifier.load_from_checkpoint(
        model_checkpoint, model=model
    )
    model = pl_module.model
    model.to(device)
    model.eval()

    return model, tokenizer, reverse_mapping, thresholds


def predict(
    text, model, tokenizer, reverse_mapping, thresholds, device, max_length=512
):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask.to(device))

    outputs = outputs.sigmoid().cpu().numpy().reshape(-1)

    return_list = []
    for label in range(len(outputs)):
        if outputs[label] >= thresholds[label]:
            return_list.append(reverse_mapping[str(label)])

    return return_list
