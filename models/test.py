#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def test_nlp(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    #data_loader = DataLoader(datatest, batch_size=args.bs, drop_last=True)
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    #hidden_test = net_g.init_hidden(args.bs)

    num_data = 0
    with torch.no_grad(): 
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            
            hidden_test = net_g.init_hidden(len(target))
            #hidden_test = repackage_hidden(hidden_test)
            log_probs, hidden_test = net_g(data.t(), hidden_test)
            
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs.t(), torch.max(target,1)[1], reduction='sum').item()
            # get the index of the max log-probability
            _, pred_label = torch.max(log_probs.t(), 1)
            correct += (pred_label == torch.max(target, 1)[1]).sum().item()
            num_data = num_data + len(target)

    test_loss = test_loss / num_data
    accuracy = 100.00 * correct / num_data
    return accuracy, test_loss
