import copy 
import os
import pickle

import torch
import torch.nn as nn

from models.GAT import GATNet
from models.GNNTrainer import GNNTrainer

import pytorch_lightning as pl

import torch_geometric.data as geom_data

from torch_geometric.datasets import Planetoid
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
import torch.nn.utils.prune as prune

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "pruned_models/GATs/")
BATCH_SIZE = 128
HIDDEN_DIM = 16
NUM_HEADS = 8
LAYERS = 1
DROPOUT = 0.6
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 150

def copy_weights_module(unpruned, pruned):
    with torch.no_grad():
        u_module, u_name = unpruned
        p_module, p_name = pruned
        
        weight_pruned = getattr(p_module, p_name)
        weight_unpruned = getattr(u_module, u_name)

        weight_pruned.copy_(weight_unpruned)

def copy_weights_list(unpruned_list, pruned_list):
    zipped = zip(unpruned_list, pruned_list)
    with torch.no_grad():
        for unpruned, pruned in zipped:
            copy_weights_module(unpruned, pruned)


def check_graph_sparsity(model):
    total = model.edge_num
    non_zeros = torch.count_nonzero(getattr(model, 'adj_mask1_train'))
    return non_zeros / total * 100
 
def check_model_sparsity(modules):

    total_weights = 0
    non_zero_wei = 0

    for module, weight in modules:
        mat = getattr(module, weight)
        tot = torch.numel(mat)
        non_zeros = torch.count_nonzero(mat)
        total_weights += tot
        non_zero_wei += non_zeros

    return non_zero_wei / total_weights * 100
    
if __name__ == '__main__':
    pl.seed_everything(42)
    dataset_name = 'Cora'
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=f'{dataset_name}')

    args = {
        'seed': 42,
        'in_dim': dataset.num_node_features,
        'out_dim': dataset.num_classes,
        'num_heads': NUM_HEADS,
        'dropout':DROPOUT,
        'hidden_dim': HIDDEN_DIM,
        'n_layers': LAYERS,
        'edge_num': 13264,
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'pruning_percent_adj': 0.1,
        'pruning_percent_wei': 0.1,
        's1': 0.0001,
        's2': 0.0001,
        'mask_epoch': 20,
        'fix_epoch': 20,
    }

    # # Create a PyTorch Lightning trainer
    # root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + dataset_name)
    # os.makedirs(root_dir, exist_ok=True)
    # trainer = pl.Trainer(
    #     default_root_dir=root_dir,
    #     callbacks=[ModelCheckpoint(mode="max", monitor="val_acc"), TQDMProgressBar(refresh_rate=0)],
    #     # accelerator="mps", devices=1,
    #     max_epochs=EPOCHS,
    #     log_every_n_steps=1,
    #     num_sanity_val_steps=0
    # )  # 0 because epoch size is 1
    # trainer.logger._default_hp_metric = None
    
    stats = {}

    model = GNNTrainer(args)
    model_copy = GNNTrainer(args)
    model_copy.load_state_dict(model.state_dict())

    parameters_to_prune = (
        (model.model.layers[0].attn_fc, 'weight'),
        (model.model.layers[1].fc, 'weight'),
        (model.model.layers[1].attn_fc, 'weight'),
        (model.model.layers[0].fc, 'weight'),
    )

    prune.l1_unstructured(model.model, name='adj_mask1_train', amount=0.00)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.00,
    )

    for imp in range(21):
        
        parameters_to_prune = (
            (model.model.layers[0].attn_fc, 'weight'),
            (model.model.layers[1].fc, 'weight'),
            (model.model.layers[1].attn_fc, 'weight'),
            (model.model.layers[0].fc, 'weight'),
        )

        model.graph_spar = check_graph_sparsity(model.model)
        model.model_spar = check_model_sparsity(parameters_to_prune)

        graph_spar, model_spar = model.graph_spar, model.model_spar
        root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + dataset_name)
        os.makedirs(root_dir, exist_ok=True)
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[
                        ModelCheckpoint(mode="max", monitor="val_acc", filename=f'pruned-iteration{imp}-{graph_spar}-{model_spar}'), 
                        TQDMProgressBar(refresh_rate=0)],
            # accelerator="mps", devices=1,
            max_epochs=EPOCHS,
            log_every_n_steps=1,
            num_sanity_val_steps=0
        )  # 0 because epoch size is 1
        trainer.logger._default_hp_metric = None

        node_data_loader = geom_data.DataLoader(dataset, batch_size=1, num_workers=8)
        trainer.fit(model, node_data_loader, node_data_loader)

        model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
        
        parameters_to_prune = (
            (model.model.layers[0].attn_fc, 'weight'),
            (model.model.layers[1].fc, 'weight'),
            (model.model.layers[1].attn_fc, 'weight'),
            (model.model.layers[0].fc, 'weight'),
        )

        test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
        print('(Pruning iteration {}) | Final test accuracy: {:.2f} | Graph sparsity: {:.2f} | Model sparsity: {:.2f}'.format(
            imp,
            test_result[0]['test_acc'],
            graph_spar,
            model_spar
        ))

        stats[(imp, graph_spar, model_spar)] = test_result

        prune.l1_unstructured(model.model, name='adj_mask1_train', amount=0.05)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.20,
        )

        # REWIND WEIGHTS
        unpruned_list = [
            (model_copy.model.layers[0].attn_fc, 'weight'),
            (model_copy.model.layers[1].fc, 'weight'),
            (model_copy.model.layers[1].attn_fc, 'weight'),
            (model_copy.model.layers[0].fc, 'weight'),
        ]

        pruned_list = [
            (model.model.layers[0].attn_fc, 'weight_orig'),
            (model.model.layers[1].fc, 'weight_orig'),
            (model.model.layers[1].attn_fc, 'weight_orig'),
            (model.model.layers[0].fc, 'weight_orig'),
        ]

        copy_weights_list(unpruned_list, pruned_list)
        

    with open('statistics/GAT_pruned.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

