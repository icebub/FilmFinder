import os

import torch
import yaml
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from filmfinder.src.datasets.MovieGenres import CustomDataset, MovieGenres
from filmfinder.src.models.BaseModel import BaseModel
from filmfinder.src.modules.BertMultiLabelClassifier import BertMultiLabelClassifier


def load_config():
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    with open(f"{abs_folder}/../train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_dataset(data_path):
    movie_dataset = MovieGenres(data_path)
    texts, labels = movie_dataset.get_dataset()
    reverse_mapping = movie_dataset.reverse_mapping
    class_weights = movie_dataset.class_weight
    num_class = len(reverse_mapping.keys())
    return texts, labels, reverse_mapping, class_weights, num_class


def get_exp_path(exp_id):
    abs_folder = os.path.dirname(os.path.abspath(__file__))
    exp_path = f"{abs_folder}/../../experiments/{exp_id}"
    return exp_path


def prepare_training(
    exp_id,
    config,
    data_path,
):
    pretrain_model = config["pretrain_model"]
    seed = config["seed"]
    (
        texts,
        labels,
        reverse_mapping,
        class_weights,
        num_class,
    ) = load_dataset(data_path)

    pl_module, tokenizer = load_pl_module(
        exp_id, pretrain_model, num_class, class_weights, config["loss_fn"]
    )

    dataset = CustomDataset(texts, labels, tokenizer, max_length=512)

    test_set_ratio = config["val_set_size"]
    val_set_ratio = config["val_set_size"]

    train_set, test_set = train_test_split(
        dataset, test_size=test_set_ratio, random_state=seed
    )
    train_set, val_set = train_test_split(
        train_set, test_size=val_set_ratio, random_state=seed
    )

    return (
        pl_module,
        tokenizer,
        train_set,
        val_set,
        test_set,
        class_weights,
        reverse_mapping,
    )


def load_pl_module(
    exp_id=None,
    pretrain_model="bert-base-uncased",
    num_class=20,
    class_weights=None,
    loss_fn="BCEWithLogitsLoss",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    model = BaseModel(pretrain_model, num_classes=num_class, freeze_bert=True)

    if exp_id:
        exp_path = get_exp_path(exp_id)
        model_checkpoint = f"{exp_path}/best_model.ckpt"
        print("Loading model from checkpoint: ", model_checkpoint)
        pl_module = BertMultiLabelClassifier.load_from_checkpoint(
            model_checkpoint,
            model=model,
            map_location=device,
        )
    else:
        print("Training from scratch")
        pl_module = BertMultiLabelClassifier(
            model,
            loss_fn=loss_fn,
            class_weight=class_weights,
        )
    return pl_module, tokenizer
