import torch

import numpy as np
import dgl
from utils_lth import load_data, load_adj_raw
from gnns.gat_net import GATNet
from plot_utils import parameters

embeddings_teacher = {
    'cora': [1433, 512, 7],
    'citeseer': [3703, 512, 6],
    'pubmed': [500, 512, 3],
}


embeddings_student = {
    'cora': [1433, 68, 7],
    'citeseer': [3703, 68, 6],
    'pubmed': [500, 68, 3],
}


for dataset in ['cora', 'citeseer', 'pubmed']:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
    adj = load_adj_raw(dataset)
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    g.add_edges(list(range(node_num)), list(range(node_num)))
    args = {
        'embedding_dim': embeddings_teacher[dataset],
        'num_heads': 8,
        'dropout': 0.6,
        'n_layers': 2,
        'residual': False,
    }

    model = GATNet(args, g, pruning=False)
    teacher_params = parameters(model)
    print(f'Teacher parameters on {dataset}: {teacher_params}')

    args = {
        'embedding_dim': embeddings_student[dataset],
        'num_heads': 2,
        'dropout': 0.6,
        'n_layers': 4,
        'residual': False,
    }
    model = GATNet(args, g, pruning=False)
    student_params = parameters(model)
    print(f'Student parameters on {dataset}: {student_params}')
