import json
import os
from datetime import datetime

import pytorch_lightning as pl
from modules.utils import get_exp_path, load_config, prepare_training
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

abs_folder = os.path.dirname(os.path.abspath(__file__))
exp_id = datetime.now().strftime("%Y%m%d%H%M")
save_path = get_exp_path(exp_id)
if not os.path.exists(save_path):
    os.makedirs(save_path)

config = load_config()

abs_folder = os.path.dirname(os.path.abspath(__file__))
data_path = f"{abs_folder}/data/movies_metadata.csv"

pretrain_model = config["pretrain_model"]
batch_size = config["batch_size"]
num_workers = config["num_workers"]
loss_fn = config["loss_fn"]
seed = config["seed"]

(
    pl_module,
    tokenizer,
    train_set,
    val_set,
    test_set,
    class_weights,
    reverse_mapping,
) = prepare_training(None, config, data_path)

num_class = len(reverse_mapping.keys())
save_label_data = {
    "reverse_mapping": reverse_mapping,
    "num_class": num_class,
    "batch_size": batch_size,
    "seed": seed,
    "pretrain_model": pretrain_model,
    "loss_fn": loss_fn,
    "class_weights": class_weights,
}
with open(f"{save_path}/label_data.json", "w") as f:
    json.dump(save_label_data, f)


early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=save_path,
    filename="best_model",
    save_top_k=1,
    mode="min",
)
logger = TensorBoardLogger("tb_logs", name="my_model")

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

trainer = pl.Trainer(
    precision=16,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    max_epochs=100,
)
trainer.fit(pl_module, train_loader, val_loader)

print("finish training")
