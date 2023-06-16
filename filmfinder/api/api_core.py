import json
import os
import pickle

import torch
import yaml
from transformers import BertTokenizer

from filmfinder.src.models.BaseModel import BaseModel
from filmfinder.src.modules.BertMultiLabelClassifier import BertMultiLabelClassifier
from filmfinder.src.modules.utils import get_exp_path


def load_config():
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    with open(f"{abs_folder}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_model(exp_id, device):
    exp_path = get_exp_path(exp_id)

    with open(f"{exp_path}/label_data.json", "r") as f:
        model_data = json.load(f)

    pretrain_model = model_data["pretrain_model"]
    reverse_mapping = model_data["reverse_mapping"]
    num_class = model_data["num_class"]

    with open(f"{exp_path}/eval_data.pkl", "rb") as f:
        eval_data = pickle.load(f)

    f1_mappping = eval_data["f1_mapping"]
    thresholds = eval_data["thresholds"]

    model, tokenizer = load_transformer_model(
        exp_path, pretrain_model, num_class, device
    )

    return model, tokenizer, reverse_mapping, thresholds, f1_mappping


def load_transformer_model(exp_path, pretrain_model, num_class, device):
    model_checkpoint = f"{exp_path}/best_model.ckpt"

    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    model = BaseModel(pretrain_model, num_classes=num_class, freeze_bert=True)

    pl_module = BertMultiLabelClassifier.load_from_checkpoint(
        model_checkpoint, model=model, map_location=device
    )
    model = pl_module.model
    model.to(device)
    model.eval()

    return model, tokenizer


def model_predict(model, input_ids, attention_mask, device):
    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask.to(device))
    return outputs.sigmoid().cpu().numpy().reshape(-1).tolist()


def predict(
    text,
    model,
    tokenizer,
    reverse_mapping,
    thresholds,
    f1_mappping,
    device,
    max_length=512,
):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    outputs = model_predict(
        model, encoding["input_ids"], encoding["attention_mask"], device
    )

    return_list = []
    for label in range(len(outputs)):
        if outputs[label] >= thresholds[label]:
            pred = str(format(outputs[label], ".3f"))
            return_list.append(
                {
                    "genre": reverse_mapping[str(label)],
                    "confidence": f1_mappping[label][pred],
                }
            )
    return_list = sorted(return_list, key=lambda k: k["confidence"], reverse=True)
    return return_list
