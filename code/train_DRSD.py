# -*- coding: utf-8 -*-
import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from shgnn import *
from utils import *
import argparse


class Trainer_DRSD(object):
    def __init__(self, args):
        super(Trainer_DRSD, self).__init__()
        self.args = args
        self.root_path = '../'
        self.log_path = self.root_path + 'log/{task}/{model}'.format(task=self.args.task, model=self.args.model)
        self.save_model_path = self.root_path + 'save_model/{task}/{model}'.format(task=self.args.task, model=self.args.model)

    def data_prepare(self):
        label = np.load(self.root_path + 'data/{task}/label_array.npy'.format(task=self.args.task))
        label = torch.from_numpy(label).float().to('cuda')
        features = np.load(self.root_path +'data/{task}/features.npy'.format(task=self.args.task))
        features = torch.from_numpy(features).float().to('cuda')
        return features, label
    
    def load_graph(self):
        (g,), _ = dgl.load_graphs(self.root_path + 'data/{task}/urban_graph.dgl'.format(task=self.args.task))
        g = g.to('cuda')
        return g

    def graph_process(self, g):
        out_degree = g.out_degrees().float().clamp(min=1)
        in_degree = g.in_degrees().float().clamp(min=1)
        g.ndata['out_degree_norm'] = torch.pow(out_degree, -0.5).view(-1, 1)
        g.ndata['in_degree_norm'] = torch.pow(in_degree, -0.5).view(-1, 1)
        return g
    
    def get_mask(self):
        with open(self.root_path + 'data/{task}/mask.json'.format(task=self.args.task), 'r') as f:
            mask_dict = json.load(f)
        return mask_dict

    def train(self, train_id, val_id, test_id, save_model, log):
        features, label = self.data_prepare()
        g = self.load_graph()
        g = self.graph_process(g)
        num_nodes = g.num_nodes()
        train_mask, val_mask, test_mask = torch.tensor([False] * num_nodes), torch.tensor([False] * num_nodes), torch.tensor([False] * num_nodes)
        train_mask[torch.tensor(train_id)] = True
        val_mask[torch.tensor(val_id)] = True
        test_mask[torch.tensor(test_id)] = True
        train_mask, val_mask, test_mask= train_mask.to('cuda'), val_mask.to('cuda'), test_mask.to('cuda')
        
        model = SHGNN_DRSD(
            task=self.args.task,
            g=g, 
            in_dim=self.args.in_dim, 
            out_dim=self.args.out_dim, 
            pool_dim=self.args.pool_dim,
            num_sect=self.args.num_sect, 
            rotation=self.args.rotation,
            head_sect=self.args.head_sect,
            num_ring=self.args.num_ring, 
            bucket_interval=self.args.bucket_interval,
            head_ring=self.args.head_ring,
            drop_rate=self.args.drop).to('cuda')
        
        param_list = [{'params':model.parameters(), 'lr':self.args.lr, 'weight_decay': self.args.decay}]
        optimizer = optim.Adam(param_list)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.999, last_epoch=-1)
        loss_fn = nn.BCELoss()

        best_epoch, best_auc = 0, 0

        for epoch in range(self.args.epoch_num+1):
            ########################## train ###########################
            model.train()
            prob = model(features)
            
            prob_train = prob[train_mask]
            label_train = label[train_mask]
            
            loss = loss_fn(prob_train, label_train)
            train_loss = loss.item()

            fpr, tpr, _ = roc_curve(label_train.tolist(), prob_train.tolist(), pos_label=1) 
            AUC_train = auc(fpr, tpr)            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            ########################## val ###########################
            model.eval()
            prob = model(features)

            prob_val = prob[val_mask]
            label_val = label[val_mask]
            fpr, tpr, _ = roc_curve(label_val.tolist(), prob_val.tolist(), pos_label=1) 
            AUC_val = auc(fpr, tpr)  

            prob_test = prob[test_mask]
            label_test = label[test_mask]
            fpr, tpr, _ = roc_curve(label_test.tolist(), prob_test.tolist(), pos_label=1) 
            AUC_test = auc(fpr, tpr)  

            log_new = 'epoch: %3d, train loss: %.6s | train AUC: %.6s, val AUC: %.6s, test AUC: %.6s' % \
                      (epoch, train_loss, AUC_train, AUC_val, AUC_test)
            log += log_new + '\n'
            print(log_new)

            if AUC_val > best_auc:
                best_auc = AUC_val
                best_epoch = epoch
                best_auc_test = AUC_test

            if epoch == save_model:
                self.save_model(model)
                break
    
        log_new = 'best epoch: %.3d | best_AUC_val: %.6s, best_AUC_test: %.6s' % (best_epoch, best_auc, best_auc_test)
        log += '\n' + log_new + '\n'
        print(log_new)
        return log, best_epoch
   
    def Train(self):
        # load the sample id of train/val/test set
        mask_dict = self.get_mask()
        train_id, val_id, test_id = mask_dict['train'], mask_dict['val'], mask_dict['test']

        seed_setup(self.args.seed)
        log = str(self.args) + '\n------------------- start training ----------------------\n'
        log, best_epoch = self.train(train_id=train_id, val_id=val_id, test_id=test_id, save_model=-1, log=log)

        # output the training log
        with open(self.log_path, 'w') as f:
            f.write(log)
        
        # retrain and save model at the epoch with best performance on validation set
        seed_setup(self.args.seed)
        log , best_epoch = self.train(train_id=train_id, val_id=val_id, test_id=test_id, save_model=best_epoch, log=log)


    def save_model(self, model):
        torch.save(model.state_dict(), self.save_model_path + '.pth')


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--epoch_num', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--decay', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default=None)
    # for SHGNN
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--out_dim', type=int, default=None)
    parser.add_argument('--pool_dim', type=int, default=None)
    parser.add_argument('--num_sect', type=int, default=None)
    parser.add_argument('--rotation', type=float, default=None)
    parser.add_argument('--head_sect', type=int, default=None)
    parser.add_argument('--num_ring', type=int, default=None)
    parser.add_argument('--bucket_interval', type=str, default=None)
    parser.add_argument('--head_ring', type=int, default=None)
    parser.add_argument('--drop', type=float, default=None)
   
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    Trainer = Trainer_DRSD(args=args)
    Trainer.Train()
