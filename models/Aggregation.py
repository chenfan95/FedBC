#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np

def FedAvgPers(w, lambs):
    w_avg = copy.deepcopy(w[0])
    #lamb_sum = torch.sum(lambs)
    lamb_sum = np.sum(lambs)
    lamb_coef = lambs/ lamb_sum
    #print(torch.sum(lamb_check))
    for k in w_avg.keys():
        w_avg[k] = lamb_coef[0] * w_avg[k]
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + lamb_coef[i] * w[i][k]
        #w_avg[k] = torch.div(w_avg[k], lamb_sum)
    return w_avg

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_sample(w, len_data):
    w_avg = copy.deepcopy(w[0])
    coef = len_data / np.sum(len_data)
    for k in w_avg.keys():
        w_avg[k] = coef[0] * w_avg[k]
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + coef[i] * w[i][k]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
