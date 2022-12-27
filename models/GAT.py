import numpy as np

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_scipy_sparse_matrix

"""
    Layer code modified from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/graphs/gat/__init__.py
"""
class GATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                is_concat: bool = True,
                dropout: float = 0.6,
                leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.fc = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn_fc = nn.Linear(self.n_hidden * 2, 1, bias=False)

        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)


    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, train_mask):
        eyes = torch.stack((torch.arange(h.shape[0]), torch.arange(h.shape[0])))
        edge_index = torch.cat((edge_index, eyes), dim=-1)
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformation,
        # $$\overrightarrow{g^k_i} = \mathbf{W}^k \overrightarrow{h_i}$$
        # for each head.
        # We do single linear transformation and then split it up for each head.
        
        g = self.fc(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn_fc(g_concat))
        e = e.squeeze(-1)

        # pruned = torch.arange(train_mask.shape[-1]).float()
        # mat = to_scipy_sparse_matrix(masked_edges)
        # adj_mat = np.reshape(mat, (mat.shape[0], mat.shape[1], 1))

        # assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        # assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        # assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        
        adj_mat = torch.zeros((n_nodes, n_nodes))
        adj_mat[edge_index[0], edge_index[1]] = train_mask
        adj_mat[range(n_nodes), range(n_nodes)] = 1.0
        adj_mat = adj_mat.unsqueeze(-1)
        
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)

class GATNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args['in_dim']
        self.out_channels = args['out_dim']
        self.num_heads = args['num_heads']
        self.dropout = args['dropout']
        self.hidden_dim = args['hidden_dim']
        self.n_layers = args['n_layers']
        self.edge_num = args['edge_num']
        self.layers = nn.ModuleList(
            [GATLayer(self.in_channels, self.hidden_dim, 
                            n_heads=self.num_heads, dropout=self.dropout) for _ in range(self.n_layers)])
        self.activation = nn.ELU()
        self.layers.append(GATLayer(self.hidden_dim, self.out_channels, n_heads=1, dropout=0.0, is_concat=False))
        self.dropout_layer = nn.Dropout(self.dropout)
        self.adj_mask1_train = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.edge_num), mean=1, std=1e-4), requires_grad=True)
        # self.adj_mask2_fixed = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.edge_num), mean=1, std=1e-4), requires_grad=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        pred = x
        for layer in self.layers[:-1]:
            # pred = layer(pred, edge_index, self.adj_mask1_train, self.adj_mask2_fixed)
            pred = layer(pred, edge_index, self.adj_mask1_train)
    
        pred = self.activation(pred)
        pred = self.dropout_layer(pred)
        # pred = self.layers[-1](pred, edge_index, self.adj_mask1_train, self.adj_mask2_fixed)
        pred = self.layers[-1](pred, edge_index, self.adj_mask1_train)

        return pred