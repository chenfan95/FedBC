import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math
from torch.utils.data import TensorDataset
import torch

#alpha = 1.0
#beta = 1.0
# iid = 1 -> True
#iid = 0

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def generate_synthetic(args):

    dimension = 60
    NUM_CLASS = 10
    NUM_USER = args.num_users
    alpha = args.alpha
    beta = args.beta
    if args.iid == True:
        iid  = 1
    else:
        iid = 0

    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    #samples_per_user = (np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50) * 5
    print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        #print(mean_x[i])

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))
    return X_split, y_split


def generate_synthetic_datasets(args, train_test_split = 0.8):
    # Create data structure
    X_train = {} 
    y_train = {}
    X_test = {} 
    y_test = {}
    trainDataset_local = {}
    testDataset_local = {}
    NUM_USER = args.num_users
    X, y = generate_synthetic(args)

    partition_stats = {}
    for i in trange(NUM_USER, ncols=120):

        #uname = 'f_{0:05d}'.format(i)        
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(train_test_split * num_samples)
        test_len = num_samples - train_len

        partition_stats[i] = len(X[i])
        X_train_i = np.array(X[i][:train_len])
        X_test_i = np.array(X[i][train_len:])
        y_train_i = np.array(y[i][:train_len])
        y_test_i = np.array(y[i][train_len:])

        if i == 0:
            trainData_X = X_train_i
            testData_X = X_test_i
            trainData_y = y_train_i
            testData_y = y_test_i
        else:
            trainData_X = np.concatenate((trainData_X, X_train_i))
            testData_X = np.concatenate((testData_X, X_test_i))
            trainData_y = np.concatenate((trainData_y, y_train_i))
            testData_y = np.concatenate((testData_y, y_test_i))

        trainDataset_local[i] = TensorDataset(torch.from_numpy(X_train_i).float(),\
                                             torch.from_numpy(y_train_i).long()) 
        testDataset_local[i] = TensorDataset(torch.from_numpy(X_test_i).float(),\
                                             torch.from_numpy(y_test_i).long()) 

    trainDataset =  TensorDataset(torch.from_numpy(trainData_X).float(),\
                                             torch.from_numpy(trainData_y).long())
    testDataset = TensorDataset(torch.from_numpy(testData_X).float(),\
                                             torch.from_numpy(testData_y).long())
    
    return trainDataset, testDataset, trainDataset_local, testDataset_local, partition_stats