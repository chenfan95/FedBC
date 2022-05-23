#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.synthetic import *
from utils.options import args_parser
from utils.language_utils import *
from models.Update import *
from models.Nets import MLP, CNNMnist, CNNCifar, logistic_regression
from models.Aggregation import FedAvg, FedAvg_sample, FedAvgPers
from models.test import test_img
from utils.partition_data import generate_power_dataset
import random
import datetime
from datetime import date
import pickle


if __name__ == '__main__':
    
    args = args_parser()

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist' or args.dataset == "cifar10":
        dataset_train, dataset_test, trainDataset_local , testDataset_local, partition_stats = generate_power_dataset(args)
    elif args.dataset == "shakespeare":
        trainDataset, testDataset, trainDataset_local, testDataset_local, data_stats = generate_shake()
    elif args.dataset == "synthetic":
        dataset_train, dataset_test, trainDataset_local , testDataset_local, partition_stats = generate_synthetic_datasets(args)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    if args.dataset == "mnist" or args.dataset == "cifar10":
        # skewness factor = 0.0 indicates iid
        output_file = 'p_{}_TN{}_sk{}_C{}_s{}'.format(args.dataset, args.num_users, args.skewness_factor,\
                                 args.classes_per_partition, args.seed)
    elif args.dataset == "synthetic":
        output_file = 'p_{}_TN{}_iid_{}_s{}'.format(args.dataset, args.num_users, args.iid, args.seed)
    print(output_file)
    
    if not os.path.isdir(args.paritions_dir):
        os.mkdir(args.paritions_dir)
    output_file =  args.paritions_dir + "/" + output_file + ".p"
    pickle.dump((dataset_train, dataset_test, trainDataset_local,testDataset_local,partition_stats), open(output_file, "wb" ))
    
