import os

from pytorch_lightning.callbacks import ModelCheckpoint

import torch_geometric.data as geom_data
import pytorch_lightning as pl
from models.GNNTrainer import GATNet

from torch_geometric.datasets import Planetoid
from pytorch_lightning.callbacks import TQDMProgressBar

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")
BATCH_SIZE = 128
HIDDEN_DIM = 16
NUM_HEADS = 8
LAYERS = 1
DROPOUT = 0.6
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 300

def train_node_classifier(model_name, dataset_name, **model_kwargs):
    pl.seed_everything(42)
    
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=f'{dataset_name}')
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1, num_workers=8)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(mode="max", monitor="val_acc"), TQDMProgressBar(refresh_rate=0)],
        # accelerator="mps", devices=1,
        max_epochs=EPOCHS,
        log_every_n_steps=1,
    )  # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model, loading...")
    #     model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    # else:
    pl.seed_everything()
    model = GATNet(
        in_dim=dataset.num_node_features, 
        hidden_dim=HIDDEN_DIM,
        out_dim=dataset.num_classes, 
        num_heads=NUM_HEADS,
        n_layers=LAYERS,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

    )
    trainer.fit(model, node_data_loader, node_data_loader)
    model = GATNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result

if __name__ == "__main__":
    # train_node_classifier('GAT_test', 'Cora') # version_1 on tensorboard
    # train_node_classifier('GAT_test', 'Citeseer') # version 4 on tensorboard
    train_node_classifier('GAT_test', 'Pubmed') # version 5 on tensorboard

