import os

import pytorch_lightning as pl
import torch
from datasets.MovieGenres import CustomDataset, MovieGenres
from models.BaseModel import BaseModel
from modules.BertMultiLabelClassifier import BertMultiLabelClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

torch.multiprocessing.set_start_method("spawn")
pretrain_model = "bert-base-uncased"

NUM_CLASSES = 32
BATCH_SIZE = 8
NUM_WORKERS = 0

tokenizer = BertTokenizer.from_pretrained(pretrain_model)
model = BaseModel(pretrain_model, num_classes=NUM_CLASSES, freeze_bert=True)

abs_folder = os.path.dirname(os.path.abspath(__file__))
data_path = f"{abs_folder}/data/movies_metadata.csv"
movie_dataset = MovieGenres(data_path)
texts, labels = movie_dataset.get_dataset()
class_mapping = movie_dataset.mapping
reverse_mapping = movie_dataset.reverse_mapping

dataset = CustomDataset(texts, labels, tokenizer, max_length=512)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

test_set_ratio = 0.1
val_set_ratio = 0.1
train_set_ratio = 1 - test_set_ratio - val_set_ratio

# train test split
train_set, test_set = train_test_split(dataset, test_size=test_set_ratio)
train_set, val_set = train_test_split(train_set, test_size=val_set_ratio)

pl_module = BertMultiLabelClassifier(model)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

trainer = pl.Trainer(precision=16)

trainer.fit(pl_module, train_loader, val_loader)
