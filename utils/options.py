#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--mu', type=float, default=0., help="mu or 1/eta")
    parser.add_argument('--gamma', type=float, default=0., help="gamma")
    parser.add_argument('--num_users', type=int, default=30, help="number of users: K")
    parser.add_argument('--num_local_users', type=float, default=10, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha")
    parser.add_argument('--beta', type=float, default=0.5, help="beta")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lamb_lr', type=float, default=0.001, help="lamb learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lamb_momentum', type=float, default=0.5, help="SGD lamb momentum (default: 0.5)")
    parser.add_argument('--q', type=float, default=0.0, help="q for qfedavg")
    parser.add_argument('--personal_lamda', type=float, default=0.01, help="personal learning rate")
    parser.add_argument('--personal_lr', type=float, default=0.01, help="personal learning rate")
    
    #parser.add_argument('--weight_decay', type=float, default=0.0, help="weight_decay for nlp")

    # MAML training 
    parser.add_argument('--K', type=int, default=1, help="maml inner update")
    parser.add_argument('--train_MAML', action='store_true', help='perform MAML-type of training')
    parser.add_argument('--test_MAML', action='store_true', help='whether to test MAML')
    parser.add_argument('--eval_one_step', action='store_true', help='perform one gradient step then evaluate')
    parser.add_argument('--inner_lr', type=float, default=0.01, help="learning rate inner")
    parser.add_argument('--outer_lr', type=float, default=0.01, help="learning rate outer")

    # model arguments
    parser.add_argument('--model', type=str, default='cifarCNN', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="mnist/cifar10/synthetic")
    parser.add_argument('--classes_per_partition', type=int, default=2, help="numer of classes per user")
    parser.add_argument('--skewness_factor', type=float, default=0.0, help="constant in power law")

    # others
    parser.add_argument('--alg', type=str, default='fedavg', help="fedavg/fedprox/fedbc/scaffold/qfedavg/pfedme")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--fix_gamma', action='store_true', help='whether to fix gamma or not')
    parser.add_argument('--eval_local', action='store_true', help='whether to evaluate local model or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--results_dir', type=str, default="Results", help='results dir')
    parser.add_argument('--paritions_dir', type=str, default="Partitions", help='partitions dir')
    parser.add_argument('--read_partition', action='store_true', help='whether to read partition from partitions dir or not')
    args = parser.parse_args()
    return args
