import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from datasets.MovieGenres import CustomDataset, MovieGenres
from models.BaseModel import BaseModel
from modules.BertMultiLabelClassifier import BertMultiLabelClassifier
from modules.loss_fn import balanced_log_loss
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

pretrain_model = "bert-base-uncased"


BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42


abs_folder = os.path.dirname(os.path.abspath(__file__))
data_path = f"{abs_folder}/data/movies_metadata.csv"
movie_dataset = MovieGenres(data_path)
texts, labels = movie_dataset.get_dataset()
class_mapping = movie_dataset.mapping
reverse_mapping = movie_dataset.reverse_mapping
class_weights = movie_dataset.class_weight

num_class = len(class_mapping)

tokenizer = BertTokenizer.from_pretrained(pretrain_model)
model = BaseModel(pretrain_model, num_classes=num_class, freeze_bert=True)

dataset = CustomDataset(texts, labels, tokenizer, max_length=512)

test_set_ratio = 0.1
val_set_ratio = 0.1
train_set_ratio = 1 - test_set_ratio - val_set_ratio

train_set, test_set = train_test_split(
    dataset, test_size=test_set_ratio, random_state=SEED
)
train_set, val_set = train_test_split(
    train_set, test_size=val_set_ratio, random_state=SEED
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
)

exp_id = "N_202306132218"
exp_path = f"{abs_folder}/../experiments/{exp_id}"

model_checkpoint = f"{exp_path}/best_model.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pl_module = BertMultiLabelClassifier.load_from_checkpoint(model_checkpoint, model=model)
model = pl_module.model
model.to(device)
model.eval()

test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

print("Evaluating on test set")
all_preds = []
all_labels = []
for idx, batch in enumerate(test_loader):
    print(idx, "/", len(test_loader))
    with torch.no_grad():
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask)
        labels = labels.cpu().numpy()
        predicted = outputs.cpu().numpy()

        all_preds.extend(predicted)
        all_labels.extend(labels)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

loss = balanced_log_loss(all_preds, all_labels, class_weights)
print("Loss: ", loss)

n_labels = all_labels.shape[1]
auc_roc_scores = []
best_threshold_list = []
f1_scores_list = []
f1_mappping = defaultdict(dict)

for label in range(n_labels):
    labels = all_labels[:, label]
    preds = all_preds[:, label]
    preds = 1 / (1 + np.exp(-preds))

    auc = metrics.roc_auc_score(labels, preds)
    auc_roc_scores.append(auc)

    thresholds = np.linspace(0, 1, 1000)
    last_precision = 0
    f1_scores = []
    for threshold in thresholds:
        labels = all_labels[:, label]
        preds = all_preds[:, label]
        preds = 1 / (1 + np.exp(-preds))
        y_pred = (preds >= threshold).astype(int)
        f1_scores.append(metrics.f1_score(labels, y_pred))

        threshold = str(format(threshold, ".3f"))
        if np.sum(y_pred) == 0:
            precision = last_precision
        else:
            precision = metrics.precision_score(labels, y_pred)
            last_precision = precision
        f1_mappping[label][threshold] = round(float(precision), 4)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)

    print("Class:", reverse_mapping[label])
    print("Best Threshold:", best_threshold)
    print("Best F1 Score:", best_f1_score)
    best_threshold_list.append(best_threshold)
    f1_scores_list.append(best_f1_score)


average_auc = round(np.mean(auc_roc_scores), 4)
average_f1 = round(np.mean(f1_scores_list), 4)
print("Average AUC: ", average_auc)
print("Average F1 Score: ", average_f1)

save_data = {
    "f1_mapping": f1_mappping,
    "thresholds": best_threshold_list,
    "average_auc": average_auc,
    "average_f1": average_f1,
    "loss": loss,
}

with open(f"{exp_path}/eval_data.pkl", "wb") as f:
    pickle.dump(save_data, f)
