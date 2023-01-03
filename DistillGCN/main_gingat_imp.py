import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

from torchsummary import summary
import net as net
from utils_lth import load_data, load_adj_raw
from sklearn.metrics import f1_score

import pickle
import dgl
from gnns.gin_net import GINNet
from gnns.gat_net import GATNet
import pruning
import pruning_gat
import pdb
import warnings
warnings.filterwarnings('ignore')
import copy

PATH = '/Users/antoniaboca/vs_code/l46_project/lth-models'
def run_fix_mask(args, imp_num, rewind_weight_mask):

    pruning.setup_seed(args['seed'])
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    loss_func = nn.CrossEntropyLoss()

    g.add_edges(list(range(node_num)), list(range(node_num)))
    net_gcn = GATNet(args, g, pruning=True)
    pruning_gat.add_mask(net_gcn)

    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
    adj_spar, wei_spar = pruning_gat.print_sparsity(net_gcn)

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}
    best_model = None

    for epoch in range(args['fix_epoch']):

        optimizer.zero_grad()
        output, _ = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output, _ = net_gcn(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
                best_val_acc['summary'] = str(summary(net_gcn, verbose=0))
                best_model = copy.deepcopy(net_gcn.state_dict())


        print("IMP[{}] (Fix Mask) Epoch:[{}/{}] LOSS:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
               .format(imp_num, epoch, 
                                args['fix_epoch'],
                                loss,
                                acc_val * 100, 
                                acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    print("syd final: [{},gat] IMP[{}] (Fix Mask) Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
                 .format(   args['dataset'],
                            imp_num,
                            best_val_acc['val_acc'] * 100, 
                            best_val_acc['test_acc'] * 100, 
                            best_val_acc['epoch'],
                            adj_spar,
                            wei_spar))
    pruned_model = {
        'args': args,
        'stats': best_val_acc,
        'state_dict': best_model,
    }

    with open(f'{PATH}/{args["dataset"]}-iteration-{imp_num}.pickle', 'wb') as handle:
        pickle.dump(pruned_model, handle)
        print(f'Saved model to {PATH}/iteration-{imp_num}.pickle')
    
    with open(f'{PATH}/statistics.txt', 'a') as handle:
        print('Iteration {} | Graph sparsity {:.2f}, model sparsity: {:.2f}, test accuracy: {:.4f} at epoch {}'
            .format(imp_num, adj_spar, wei_spar, best_val_acc['test_acc'], best_val_acc['epoch']), file=handle)
    return best_model


def run_get_mask(args, imp_num, rewind_weight_mask=None):

    pruning.setup_seed(args['seed'])
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    
    g.add_edges(adj.row, adj.col)
    loss_func = nn.CrossEntropyLoss()

    g.add_edges(list(range(node_num)), list(range(node_num)))
    net_gcn = GATNet(args, g, pruning=True)
    pruning_gat.add_mask(net_gcn)
    
    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
    
    pruning_gat.add_trainable_mask_noise(net_gcn, c=1e-5)
    adj_spar, wei_spar = pruning_gat.print_sparsity(net_gcn)

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}

    rewind_weight = copy.deepcopy(net_gcn.state_dict())

    for epoch in range(args['mask_epoch']):

        optimizer.zero_grad()
        output, _ = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        pruning_gat.subgradient_update_mask(net_gcn, args) # l1 norm
            
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output, _ = net_gcn(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

                rewind_weight, adj_spar, wei_spar = pruning_gat.get_final_mask_epoch(net_gcn, rewind_weight, args)
                 
        print("IMP[{}] (Get Mask) Epoch:[{}/{}] LOSS:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
               .format(imp_num, epoch, 
                                args['mask_epoch'],
                                loss,
                                acc_val * 100, 
                                acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch'],
                                adj_spar,
                                wei_spar))

    return rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--mask_epoch', type=int, default=200)
    parser.add_argument('--fix_epoch', type=int, default=200)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--residual', action='store_true')
    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    if args['dataset'] == '':
        raise Exception("Plase provide a dataset: [Cora, Citeseer, Pubmed]")
    
    rewind_weight = run_fix_mask(args, 0, None)
    for imp in range(1, 21):
        
        rewind_weight = run_get_mask(args, imp, rewind_weight)
        _ = run_fix_mask(args, imp, rewind_weight)

"""
Default command line for Cora:

    python3 -u main_gingat_imp.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 150 --fix_epoch 150 --s1 1e-3 --s2 1e-3 
"""
    