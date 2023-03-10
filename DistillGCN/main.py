import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
import pickle
import dgl
import argparse
from gat import GAT
from utils import evaluate, collate
from utils import get_data_loader_small, save_checkpoint, load_checkpoint
from utils import evaluate_model, test_model, generate_label, evaluate_model_small, test_model_small
from auxilary_loss import gen_fit_loss, optimizing, gen_mi_loss, loss_fn_kd, gen_att_loss
from auxilary_model import collect_model
from auxilary_optimizer import block_optimizer
from plot_utils import loss_logger, parameters
import time
import matplotlib.pyplot as plt
import collections
import random
import pruning_gat
from gnns.gat_net import GATNet
from main_gingat_imp import run_fix_mask, run_get_mask

torch.set_num_threads(1)
PATH = '/Users/antoniaboca/vs_code/l46_project/kd-models'
def train_student(args, auxiliary_model, data, data_info, device, pruning_iteration=None, teacher_acc=None):
    '''
    mode:
        full:     training student without teacher
        mi:     training student use pseudo label and mutual information of middle layers 
        kd:     training student using classic knowledge distillation
    
    args: 
        auxiliary_model - dict
            {
                "model_name": {'model','optimizer','epoch_num'}
            }
    '''
    mode = args.mode
    best_score = 0
    best_loss = 1000.0

    # multi class loss function
    # loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_fcn = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()

    t_model = auxiliary_model['t_model']['model']
    s_model = auxiliary_model['s_model']['model']
    
    has_run = False
    g, features, labels = data_info['g'], data['features'], data['labels']
    idx_train, idx_val, idx_test = data['idx_train'], data['idx_val'], data['idx_test']
    subgraph = g.subgraph(idx_train)

    for epoch in range(args.s_epochs):
        s_model.train()
        loss_list = []
        additional_loss_list = []
        t0 = time.time()

        s_model.g = g
        for layer in s_model.layers:
            layer.g = g
            
        logits, middle_feats_s = s_model(g, features, 0, 0)
        
        # if epoch >= args.tofull:
        #     args.mode = 'full'

        additional_loss = 0
        if args.mode != 'full': # use a teacher
            logits_t = generate_label(t_model, g, features, device)

        ce_loss = loss_fcn(logits[idx_train], labels[idx_train])

        if args.mode=='full':
            loss = ce_loss
        
        elif args.mode == 'kd':
            soft_targets = F.log_softmax(logits_t[idx_train] / 10, dim=-1)
            soft_prob = F.log_softmax(logits[idx_train] / 10, dim=-1)
            soft_targets_loss = nn.KLDivLoss(log_target=True)(soft_prob, soft_targets)
            additional_loss = 100. * soft_targets_loss
            kd_loss = 100. * soft_targets_loss + 0.5 * ce_loss
            loss = kd_loss
            
        elif args.mode == 'mi':
            mi_loss = gen_mi_loss(auxiliary_model, middle_feats_s[args.target_layer], g, features, 
                                    subgraph, idx_train, device)
            additional_loss = mi_loss * args.loss_weight
            loss = ce_loss + additional_loss
        
        #optimizing(auxiliary_model, loss, ['s_model', 'local_model', 'local_model_s'])
        optimizing(auxiliary_model, loss, ['s_model'])
        loss_list.append(loss.item())
        additional_loss_list.append(additional_loss.item() if additional_loss!=0 else 0)

        loss_data = np.array(loss_list).mean()
        additional_loss_data = np.array(additional_loss_list).mean()
        print(f"Epoch {epoch:05d} | Loss: {loss_data:.4f} | Mi: {additional_loss_data:.4f} | Time: {time.time()-t0:.4f}s")
        if epoch % 10 == 0:
            score = evaluate_model_small(g, features, labels, idx_val, device, s_model, loss_fcn)
            if score > best_score:
                best_score = score
                best_loss = loss_data
                test_score = test_model_small(g, features, labels, idx_test, s_model, device, loss_fcn)
                best_model = copy.deepcopy(s_model.state_dict())
    print(f"f1 score on testset: {test_score:.4f}")
    dict = {
        'training mode': mode,
        'pruning iteration': pruning_iteration,
        'teacher accuracy': teacher_acc,
        'f1 score on test set': test_score,
        'loss on training set': best_loss,
    }
    with open(f'{PATH}/{args.dataset}/statistics.txt', 'a+') as handle:
        print('{', file=handle)
        for key, value in dict.items():
            print(f'{key}: {value}', file=handle)
        print('}', file=handle)
    
    dict['best model'] = best_model
    with open(f'{PATH}/{args.dataset}/{mode}-{pruning_iteration}', 'wb') as handle:
        pickle.dump(dict, handle)
    dict['best model'] = None
    return dict


def train_teacher(args, model, data, device):
    train_dataloader, valid_dataloader, test_dataloader, _ = data
    
    best_model = None
    best_val = 0
    # define loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.t_epochs):
        model.train()
        loss_list = []
        for batch, batch_data in enumerate(train_dataloader):
            subgraph, feats, labels = batch_data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print(f"Epoch {epoch + 1:05d} | Loss: {loss_data:.4f}")
        if epoch % 10 == 0:
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            print(f"F1-Score on valset  :        {mean_score:.4f} ")
            if mean_score > best_val:
                best_model = copy.deepcopy(model)

            train_score_list = []
            for batch, train_data in enumerate(train_dataloader):
                subgraph, feats, labels = train_data
                feats = feats.to(device)
                labels = labels.to(device)
                train_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
            print(f"F1-Score on trainset:        {np.array(train_score_list).mean():.4f}")

    # model = best_model

    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
    print(f"F1-Score on testset:        {np.array(test_score_list).mean():.4f}")
    


def main(args):
    PATH_TO_TEACHER = f'/Users/antoniaboca/vs_code/l46_project/lth-models/{args.dataset}-iteration-{args.iteration}.pickle'
    
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    # data, data_info = get_data_loader(args)
    features, labels, idx_train, idx_val, idx_test, data_info = get_data_loader_small(dataset=args.dataset)
    
    data = {
        'features': features,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }
    
    with open(PATH_TO_TEACHER, 'rb') as handle:
        info = pickle.load(handle)
        stats = info['stats']
        t_args = info['args']
        state_dict = info['state_dict']
    
    model_dict = collect_model(args, t_args, data_info)
    t_model = model_dict['t_model']['model']
    t_model.load_state_dict(state_dict)
    
    print(f'Loaded teacher. Teacher test accuracy: {stats["test_acc"]}')


    # load or train the teacher
    # if os.path.isfile("./models/t_model.pt"):
    #     load_checkpoint(t_model, "./models/t_model.pt", device)
    # else:
    #     print("############ train teacher #############")
    #     train_teacher(args, t_model, data, device)
    #     save_checkpoint(t_model, "./models/t_model.pt")
    

    print(f"number of parameter for teacher model: {parameters(t_model)}")
    print(f"number of parameter for student model: {parameters(model_dict['s_model']['model'])}")

    # verify the teacher model
    # loss_fcn = torch.nn.BCEWithLogitsLoss()
    # train_dataloader, _, test_dataloader, _ = data
    # print(f"test acc of teacher:")
    # test_model(test_dataloader, t_model, device, loss_fcn)
    # print(f"train acc of teacher:")
    # test_model(train_dataloader, t_model, device, loss_fcn)
    
    return train_student(args, model_dict, data, data_info, device, args.iteration, stats['test_acc'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--all', action='store_true')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--iteration", type=int, default=0)

    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")


    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")

    parser.add_argument("--t-epochs", type=int, default=60,
                        help="number of training epochs")
    parser.add_argument("--t-num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--t-num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--t-num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--t-num-hidden", type=int, default=256,
                        help="number of hidden units")

    parser.add_argument("--s-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--s-num-heads", type=int, default=2,
                        help="number of hidden attention heads")
    parser.add_argument("--s-num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--s-num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--s-num-hidden", type=int, default=68,
                        help="number of hidden units")
    parser.add_argument("--target-layer", type=int, default=2,
                        help="the layer of student to learn")
    
    parser.add_argument("--mode", type=str, default='mi')
    parser.add_argument("--train-mode", type=str, default='together',
                        help="training mode: together, warmup")
    parser.add_argument("--warmup-epoch", type=int, default=600,
                        help="steps to warmup")
    
    parser.add_argument('--loss-weight', type=float, default=1.0,
                        help="weight coeff of additional loss")
    parser.add_argument('--seed', type=int, default=100,
                        help="seed")
    parser.add_argument('--tofull', type=int, default=30,
                        help="change mode to full after tofull epochs")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.mode = 'full'
    args.iteration = 0
    print(main(args))


    # if args.all:
    #     print(f'??????????????????????????????????????????????????? DATASET: {args.dataset} ???????????????????????????????????????????????????')
    #     args.mode = 'full'
    #     args.iteration = 0
    #     print(f'#### Train student without supervision ####')
    #     res = main(args)

    #     stats = {
    #         'full': [res],
    #         'kd': [],
    #         'mi': [],
    #     }

    #     for mode in ['kd', 'mi']:
    #         for iteration in range(21):
    #             args.iteration = iteration
    #             args.mode = mode
    #             print(f'#### Train student with {args.mode} loss with teacher at pruning iteration {args.iteration} ####')
    #             res = main(args)
    #             stats[mode].append(res)

    #     with open(f'{PATH}/{args.dataset}-3/final_results.pickle', 'wb') as handle:
    #         pickle.dump(stats, file=handle)
