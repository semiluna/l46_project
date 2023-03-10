import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from gnns.gat_layer import GATLayer
from gnns.mlp_readout_layer import MLPReadout
import pdb

class GATNet(nn.Module):

    def __init__(self, net_params, graph, pruning=False):
        super().__init__()

        in_dim_node = (net_params['embedding_dim'])[0] # node_dim (feat is an integer)
        hidden_dim = net_params['embedding_dim'][1]
        out_dim = net_params['embedding_dim'][2]
        n_classes = net_params['embedding_dim'][2]
        num_heads = net_params['num_heads']
        dropout = net_params['dropout']
        n_layers = net_params['n_layers']
        self.edge_num = graph.number_of_edges()
        self.graph_norm = False
        self.batch_norm = False
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList([GATLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual)])
        for _ in range(1, n_layers-1):
            self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, 
                                                dropout, self.graph_norm, self.batch_norm, self.residual))

        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))

        if pruning:
            self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        else:
            self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)

    def forward(self, g, h, snorm_n, snorm_e):
        middle_feats = []
        # GAT
        for conv in self.layers:
            h, middle_feat = conv(g, h, snorm_n, self.adj_mask1_train, self.adj_mask2_fixed)
            middle_feats.append(middle_feat)

        return h, middle_feats
    

class GATNet_ss(nn.Module):

    def __init__(self, net_params, num_par):
        super().__init__()

        in_dim_node = net_params[0] # node_dim (feat is an integer)
        hidden_dim = net_params[1]
        out_dim = net_params[2]
        n_classes = net_params[2]
        num_heads = 8
        dropout = 0.6
        n_layers = 1

        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList([GATLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))
        self.classifier_ss = nn.Linear(hidden_dim * num_heads, num_par, bias=False)

    def forward(self, g, h, snorm_n, snorm_e):
        # GAT
        for conv in self.layers:
            h_ss = h
            h, middle_feat = conv(g, h, snorm_n)

        h_ss = self.classifier_ss(h_ss)

        return h, h_ss
 
