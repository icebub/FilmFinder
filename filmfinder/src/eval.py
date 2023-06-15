import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from filmfinder.src.modules.loss_fn import balanced_log_loss
from filmfinder.src.modules.utils import get_exp_path, load_config, prepare_training

config = load_config()

abs_folder = os.path.dirname(os.path.abspath(__file__))
data_path = f"{abs_folder}/data/movies_metadata.csv"

batch_size = config["batch_size"]
num_workers = config["num_workers"]

assert len(sys.argv) == 2, "Please provide experiment id"
exp_id = sys.argv[1]

(
    pl_module,
    tokenizer,
    train_set,
    val_set,
    test_set,
    class_weights,
    reverse_mapping,
) = prepare_training(exp_id, config, data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = pl_module.model
model.to(device)
model.eval()

test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
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

    last_precision = 0
    f1_scores = []
    thresholds = []
    for threshold in range(1001):
        threshold = threshold / 1000

        labels = all_labels[:, label]
        preds = all_preds[:, label]
        preds = 1 / (1 + np.exp(-preds))
        y_pred = (preds >= threshold).astype(int)
        f1_scores.append(metrics.f1_score(labels, y_pred))

        threshold = str(format(threshold, ".3f"))
        thresholds.append(threshold)
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
    best_threshold_list.append(float(best_threshold))
    f1_scores_list.append(best_f1_score)


average_auc = round(np.mean(auc_roc_scores), 4)
average_f1 = round(np.mean(f1_scores_list), 4)

save_data = {
    "f1_mapping": f1_mappping,
    "thresholds": best_threshold_list,
    "average_auc": average_auc,
    "average_f1": average_f1,
    "loss": loss,
}

print("Average AUC:", average_auc)
print("Average F1:", average_f1)

exp_path = get_exp_path(exp_id)
with open(f"{exp_path}/eval_data.pkl", "wb") as f:
    pickle.dump(save_data, f)
