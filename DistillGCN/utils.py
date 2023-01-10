import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import f1_score
import dgl
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
from gat import GAT, GCN
from gnns.gat_net import GATNet
from utils_lth import load_data, load_adj_raw
import pruning_gat

def evaluate(feats, model, subgraph, labels, loss_fcn, idx_val=None):
    model.eval()
    with torch.no_grad():
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output, _ = model(subgraph, feats, 0, 0)
        loss_data = loss_fcn(output[idx_val], labels[idx_val])
        predict = output[idx_val].cpu().numpy().argmax(axis=1)
        score = f1_score(labels[idx_val].data.cpu().numpy(),
                         predict, average='micro')
    model.train()
    
    return score, loss_data.item()

def test_model_small(g, features, labels, idx_test, model, device, loss_fcn):
    test_score_list = []
    model.eval()
    with torch.no_grad():
        test_score_list.append(evaluate(features, model, g, labels, loss_fcn, idx_test)[0])
    mean_score = np.array(test_score_list).mean()
    print(f"F1-Score on testset:        {mean_score:.4f}")
    model.train()
    return mean_score

def test_model(test_dataloader, model, device, loss_fcn):
    test_score_list = []
    model.eval()
    with torch.no_grad():
        for batch, test_data in enumerate(test_dataloader):
            subgraph, feats, labels = test_data
            feats = feats.to(device)
            labels = labels.to(device)
            test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
        mean_score = np.array(test_score_list).mean()
        print(f"F1-Score on testset:        {mean_score:.4f}")
    model.train()
    return mean_score


def generate_label(t_model, subgraph, feats, device):
    '''generate pseudo lables given a teacher model
    '''
    # t_model.to(device)
    t_model.eval()
    with torch.no_grad():
        t_model.g = subgraph
        for layer in t_model.layers:
            layer.g = subgraph
        # soft labels
        logits_t, _ = t_model(subgraph, feats.float(), 0, 0)
        #pseudo_labels = torch.where(t_logits>0.5, 
        #                            torch.ones(t_logits.shape).to(device), 
        #                            torch.zeros(t_logits.shape).to(device))
        #labels = logits_t 
    return logits_t.detach()
    
def evaluate_model_small(g, features, labels, idx_val, device, s_model, loss_fcn):
    score_list = []
    val_loss_list = []
    s_model.eval()
    with torch.no_grad():
        score, val_loss = evaluate(features.float(), s_model, g, labels, loss_fcn, idx_val)
        score_list.append(score)
        val_loss_list.append(val_loss)
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    print(f"F1-Score on valset  :        {mean_score:.4f} ")
    s_model.train()
    return mean_score

def evaluate_model(valid_dataloader, train_dataloader, device, s_model, loss_fcn):
    score_list = []
    val_loss_list = []
    s_model.eval()
    with torch.no_grad():
        for batch, valid_data in enumerate(valid_dataloader):
            subgraph, feats, labels = valid_data
            feats = feats.to(device)
            labels = labels.to(device)
            score, val_loss = evaluate(feats.float(), s_model, subgraph, labels.float(), loss_fcn)
            score_list.append(score)
            val_loss_list.append(val_loss)
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    print(f"F1-Score on valset  :        {mean_score:.4f} ")
    s_model.train()
    return mean_score

    """
    train_score_list = []
    for batch, train_data in enumerate(train_dataloader):
        subgraph, feats, labels = train_data
        feats = feats.to(device)
        labels = labels.to(device)
        train_score_list.append(evaluate(feats, s_model, subgraph, labels.float(), loss_fcn)[0])
    print(f"F1-Score on trainset:        {np.array(train_score_list).mean():.4f}")
    """

def collate(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def collate_w_gk(sample):
    '''
    collate with graph_khop
    '''
    graphs, feats, labels, graphs_gk =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    graph_gk = dgl.batch(graphs_gk)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels, graph_gk


def get_teacher(args, data_info):
    '''args holds the common arguments
    data_info holds some special arugments
    '''
    # heads = ([args.t_num_heads] * args.t_num_layers) + [args.t_num_out_heads]
    # model = GAT(data_info['g'],
    #         args.t_num_layers,
    #         data_info['num_feats'],
    #         args.t_num_hidden,
    #         data_info['n_classes'],
    #         heads,
    #         F.elu,
    #         args.in_drop,
    #         args.attn_drop,
    #         args.alpha,
    #         args.residual)
    t_model = GATNet(args, data_info['g'], pruning=(args['iteration'] != 0))
    if args['iteration'] != 0:
        pruning_gat.add_mask(t_model)
    
    return t_model
    
def get_student(args, data_info):
    '''args holds the common arguments
    data_info holds some special arugments
    '''
    # heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]
    # model = GAT(data_info['g'],
    #         args.s_num_layers,
    #         data_info['num_feats'],
    #         args.s_num_hidden,
    #         data_info['n_classes'],
    #         heads,
    #         F.elu,
    #         args.in_drop,
    #         args.attn_drop,
    #         args.alpha,
    #         args.residual)

    model_args = {
        'n_layers': args.s_num_layers,
        'embedding_dim': [data_info['num_feats'], args.s_num_hidden, data_info['n_classes']],
        'num_heads': args.s_num_heads,
        'dropout': args.attn_drop,
        'residual':args.residual,
    }
    
    model = GATNet(model_args, data_info['g'])
    return model

def get_feat_info(args):
    feat_info = {}
    feat_info['s_feat'] = [args.s_num_heads*args.s_num_hidden] * args.s_num_layers
    feat_info['t_feat'] = [args.t_num_heads*args.t_num_hidden] * args.t_num_layers
    #assert len(feat_info['s_feat']) == len(feat_info['t_feat']),"number of hidden layer for teacher and student are not equal"
    return feat_info


def get_data_loader_small(dataset='cora'):
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
    adj = load_adj_raw(dataset)

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1
    feature_num = features.size()[1]

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    g.add_edges(list(range(node_num)), list(range(node_num)))

    data_info = {}
    data_info['n_classes'] = class_num
    data_info['num_feats'] = feature_num
    data_info['g'] = g
    return features, labels, idx_train, idx_val, idx_test, data_info

def get_data_loader(args):
    '''create the dataset
    return 
        three dataloders and data_info
    '''
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)
    fixed_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)

    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    data_info = {}
    data_info['n_classes'] = n_classes
    data_info['num_feats'] = num_feats
    data_info['g'] = g
    return (train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader), data_info


def save_checkpoint(model, path):
    '''Saves model
    '''
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")

def load_checkpoint(model, path, device):
    '''load model
    '''
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")

