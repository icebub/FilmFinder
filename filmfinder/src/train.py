import os

import pytorch_lightning as pl
import torch
from datasets.MovieGenres import CustomDataset, MovieGenres
from models.BaseModel import BaseModel
from modules.BertMultiLabelClassifier import BertMultiLabelClassifier
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

pretrain_model = "bert-base-uncased"


BATCH_SIZE = 8
NUM_WORKERS = 0


abs_folder = os.path.dirname(os.path.abspath(__file__))
data_path = f"{abs_folder}/data/movies_metadata.csv"
movie_dataset = MovieGenres(data_path)
texts, labels = movie_dataset.get_dataset()
class_mapping = movie_dataset.mapping
reverse_mapping = movie_dataset.reverse_mapping

num_class = len(class_mapping)

tokenizer = BertTokenizer.from_pretrained(pretrain_model)
model = BaseModel(pretrain_model, num_classes=num_class, freeze_bert=True)


dataset = CustomDataset(texts, labels, tokenizer, max_length=512)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

test_set_ratio = 0.1
val_set_ratio = 0.1
train_set_ratio = 1 - test_set_ratio - val_set_ratio

train_set, test_set = train_test_split(dataset, test_size=test_set_ratio)
train_set, val_set = train_test_split(train_set, test_size=val_set_ratio)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="best_model",
    save_top_k=1,
    mode="min",
)

pl_module = BertMultiLabelClassifier(model)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

trainer = pl.Trainer(precision=16, callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(pl_module, train_loader, val_loader)

print("Done")
