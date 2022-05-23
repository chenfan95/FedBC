#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
#from sklearn import metrics
from torch.optim import Optimizer
import copy
import torch.nn.functional as F
from utils.language_utils import *
from models.Optimizers import pFedMeOptimizer

class MySGD(Optimizer):
    def __init__(self, params, lr, momentum):
        defaults = dict(lr=lr, momentum = momentum)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class evaluate_local(object):
    def __init__(self, args, model, test_dataset):
        #self.total_test_samples = len(test_dataset)
        #self.trainloader = DataLoader(train_dataset, args.bs)
        #self.test_dataset = test_dataset
        self.testloader = DataLoader(test_dataset, args.local_bs)
        self.iter_test = iter(self.testloader)
        self.model = model
        self.args = args

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_test)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_test = iter(self.testloader)
            (X, y) = next(self.iter_test)
        return (X.to(self.args.device), y.to(self.args.device))

    def eval_nlp(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        l = len(data_loader)
        #hidden_test = self.model.init_hidden(self.args.bs)
        num_data = 0 
        with torch.no_grad(): 
            for idx, (x, y) in enumerate(data_loader):
                if self.args.gpu != -1:
                    x, y = x.cuda(), y.cuda()
                
                hidden_test = self.model.init_hidden(len(y))
                #hidden_test = repackage_hidden(hidden_test)
                log_probs, hidden_test = self.model(x.t(), hidden_test)
                
                # sum up batch loss
                total_loss += F.cross_entropy(log_probs.t(), torch.max(y,1)[1], reduction='sum').item()
                # get the index of the max log-probability
                _, pred_label = torch.max(log_probs.t(), 1)
                total_correct += (pred_label == torch.max(y, 1)[1]).sum().item()
                num_data = num_data + y.shape[0]
        return total_loss, total_correct, num_data


    def eval(self, data_loader):
        self.model.eval()
        # testing
        total_loss = 0
        total_correct = 0
        num_data = 0 
        l = len(data_loader)
        for idx, (x, y) in enumerate(data_loader):
            if self.args.gpu != -1:
                x, y = x.cuda(), y.cuda()
            log_probs = self.model(x)
            # sum up batch loss
            total_loss += F.cross_entropy(log_probs, y, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            total_correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
            num_data = num_data + y.shape[0]
        return total_loss, total_correct, num_data

    def train_one_step(self):
        opt_net = MySGD(self.model.parameters(), self.args.inner_lr, self.args.momentum)
        self.model.train()
        X, y = self.get_next_test_batch()
        opt_net.zero_grad()
        output = self.model(X)
        loss = F.cross_entropy(output, y)        
        # if self.args.
        
        #     loss_per = 0.
        #     for w_g, w_l in zip(glob_net.parameters(), net.parameters()):
        #         loss_per = loss_per + ((w_g - w_l).square().sum()) 
        #     loss_per = lamb * (loss_per - gamma)
        #     loss = loss + loss_per

        loss.backward()
        opt_net.step()

class LocalUpdate(object):
    def __init__(self, args, train_dataset=None, test_dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        if self.args.dataset == "shakespeare" or self.args.dataset == "sent140":
            #self.ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True, drop_last=True)
            self.ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.total_num_iters = len(self.ldr_train)
        self.iter_train = iter(self.ldr_train) 

    def evaluate_loss(self, net_g, datatest):
        net_g.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        l = len(data_loader)
        #hidden_train = net.init_hidden(self.args.local_bs)
        for idx, (x, y) in enumerate(data_loader):
            x, y = x.to(self.args.device), y.to(self.args.device)

            if self.args.dataset == "shakespeare" or self.args.dataset == "sent140":            
                hidden_train = net_g.init_hidden(len(y))
                log_probs, hidden_train = net_g(x.t(), hidden_train)
                test_loss += F.cross_entropy(log_probs.t(), torch.max(y, 1)[1], reduction='sum').item()
            else:
                log_probs = net_g(x)
                test_loss += F.cross_entropy(log_probs, y, reduction='sum').item()

        test_loss /= len(data_loader.dataset)
        return test_loss


    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_train)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_train = iter(self.ldr_train)
            (X, y) = next(self.iter_train)
        return (X.to(self.args.device), y.to(self.args.device))

    def train(self, net, glob_net, lamb, gamma):
        opt_net = torch.optim.SGD(net.parameters(), lr=self.args.lr,  momentum = self.args.momentum)
        
        for iter in range(self.args.local_ep):
            net.train()
            #hidden_train = net.init_hidden(self.args.local_bs)
            for batch_idx, (x, y) in enumerate(self.ldr_train):
                x, y = x.to(self.args.device), y.to(self.args.device)
                opt_net.zero_grad()
                
                if self.args.dataset == "shakespeare" or self.args.dataset == "sent140":
                    hidden_train = net.init_hidden(len(y))
                    #hidden_train = repackage_hidden(hidden_train)
                    log_probs, hidden_train = net(x.t(), hidden_train)
                    loss = self.loss_func(log_probs.t(), torch.max(y, 1)[1])
                else:
                    log_probs = net(x)
                    loss = self.loss_func(log_probs,y)

                if self.args.alg == 'fedprox':
                    loss_prox = 0.
                    for w_g, w_l in zip(glob_net.parameters(), net.parameters()):
                        loss_prox = loss_prox + ((w_g - w_l).square().sum()) 
                    loss_prox = self.args.mu / 2.  * loss_prox
                    loss = loss + loss_prox
                    loss.backward()
                
                elif self.args.alg == "fedavg":
                    loss.backward()

                elif self.args.alg == "fedbc":
                    loss_per = 0.
                    for w_g, w_l in zip(glob_net.parameters(), net.parameters()):
                        loss_per = loss_per + ((w_g - w_l).square().sum()) 
                    loss_per = lamb * (loss_per - gamma)
                    loss = loss + loss_per
                    loss.backward()
                opt_net.step()
        return net.state_dict()

    def train_MAML(self, net, glob_net, lamb, gamma):
        
        opt_net = MySGD(net.parameters(), self.args.inner_lr, self.args.momentum)
        for iter in range(self.args.local_ep):
            net.train()
            #hidden_train = net.init_hidden(self.args.local_bs)
            for tt in range(self.total_num_iters):
                temp_model = copy.deepcopy(list(net.parameters()))
                #step 1
                for kk in range(int(self.args.K)):
                    x, y = self.get_next_train_batch()
                    opt_net.zero_grad()
                    #hidden_train = repackage_hidden(hidden_train)
                    if self.args.dataset == "shakespeare" or self.args.dataset == "sent140":
                        hidden_train = net.init_hidden(len(y))
                        log_probs, hidden_train = net(x.t(), hidden_train)
                        loss = self.loss_func(log_probs.t(), torch.max(y, 1)[1])
                    else:
                        log_probs = net(x)
                        loss = self.loss_func(log_probs, y)

                    if self.args.alg == "fedbc":
                        loss_per = 0.
                        for w_g, w_l in zip(glob_net.parameters(), net.parameters()):
                            loss_per = loss_per + ((w_g - w_l).square().sum()) 
                        loss_per = lamb * (loss_per - gamma)
                        loss = loss + loss_per
                    loss.backward()
                    opt_net.step()

                #step 2
                x, y = self.get_next_train_batch()
                opt_net.zero_grad()
                if self.args.dataset == "shakespeare" or self.args.dataset == "sent140":
                    hidden_train = net.init_hidden(len(y))
                    log_probs, hidden_train = net(x.t(), hidden_train)
                    loss = self.loss_func(log_probs.t(), torch.max(y, 1)[1])

                else:
                    log_probs = net(x)
                    loss = self.loss_func(log_probs, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for p1, p2 in zip(net.parameters(), temp_model):
                    p1.data = p2.data.clone()
                opt_net.step(self.args.outer_lr)
            
        return net.state_dict()
  
    def train_qfedavg(self, net, glob_net):
        opt_net = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum = self.args.momentum)

        for iter in range(self.args.local_ep):
            net.train()
            #hidden_train = net.init_hidden(self.args.local_bs)
            for batch_idx, (x, y) in enumerate(self.ldr_train):
                x, y = x.to(self.args.device), y.to(self.args.device)
                opt_net.zero_grad()
                #hidden_train = repackage_hidden(hidden_train)
                if self.args.dataset == "shakespeare" or self.args.dataset == "sent140":
                    hidden_train = net.init_hidden(len(y))
                    log_probs, hidden_train = net(x.t(), hidden_train)
                    loss = self.loss_func(log_probs.t(), torch.max(y, 1)[1])
                else:
                    log_probs = net(x)
                    loss = self.loss_func(log_probs,y)
                loss.backward()
                opt_net.step()

        delta_w = []
        w_local = copy.deepcopy(list(net.parameters()))
        w_global = copy.deepcopy(list(glob_net.parameters()))
        for wg, wl in zip(w_global, w_local):
            wd = wg.data - wl.data
            delta_w.append(wd)
        return w_local, delta_w
    
    def train_scaffold(self, net, glob_net, c_local, c_global):

        opt_net = torch.optim.SGD(net.parameters(), lr=self.args.lr,  momentum = self.args.momentum)
        cnt = 0
        for iter in range(self.args.local_ep):
            net.train()
            #hidden_train = net.init_hidden(self.args.local_bs)
            for batch_idx, (x, y) in enumerate(self.ldr_train):
                x, y = x.to(self.args.device), y.to(self.args.device)
                opt_net.zero_grad()
                log_probs = net(x)
                loss = self.loss_func(log_probs, y)
                loss.backward()
                opt_net.step()
                for i,w in enumerate(net.parameters()):
                    w.data = w.data - self.args.lr * (c_global[i] - c_local[i])
                cnt += 1
            
        delta_c = []
        for i,(wg, wl) in enumerate(zip(glob_net.parameters(), net.parameters())):
            wd = wg.data - wl.data
            c_temp = c_local[i] - c_global[i] + wd / (cnt * self.args.lr)
            delta_c.append(c_temp - c_local[i])
            c_local[i] = c_temp
        return net.state_dict(), c_local, delta_c

    def train_pfedme(self, net, net_temp, lamb, gamma):
        LOSS = 0
        net.train()
        local_model = list(net_temp.parameters())
        opt_net = pFedMeOptimizer(net.parameters(), lr=self.args.personal_lr, lamda=self.args.personal_lamda)

        for iter in range(self.args.local_ep):
            for tt in range(self.total_num_iters):
        
                X, y = self.get_next_train_batch()
                for i in range(self.args.K):
                    opt_net.zero_grad()
                    output = net(X)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    persionalized_model_bar, _ = opt_net.step(local_model)

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(persionalized_model_bar, local_model):
                    localweight.data = localweight.data - self.args.personal_lamda * self.args.lr * (localweight.data - new_param.data)

        #update local model as local_weight_upated
        self.update_parameters(net, local_model)
        return net.state_dict()

    def update_parameters(self, net, new_params):
        for param , new_param in zip(net.parameters(), new_params):
            param.data = new_param.data.clone()