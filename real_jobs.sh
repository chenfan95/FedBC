#!/bin/sh

# FedAvg
python main_fed.py --eval_local --alg fedavg --lr 0.01 --epochs 5 --skewness_factor 1.2 --classes_per_partition 3 --num_users 20 --local_bs 20 --bs 128 --seed 1 --dataset cifar10 --model cifarCNN 
# q-FedAvg
python main_fed.py --eval_local --alg qfedavg --lr 0.01 --q 1.0 --epochs 5 --skewness_factor 1.2 --classes_per_partition 3 --num_users 20 --local_bs 20 --bs 128 --seed 1 --dataset cifar10 --model cifarCNN 
# FedProx
python main_fed.py --eval_local --alg qfedavg --lr 0.01 --q 1.0 --epochs 5 --skewness_factor 1.2 --classes_per_partition 3 --num_users 20 --local_bs 20 --bs 128 --seed 1 --dataset cifar10 --model cifarCNN 

# Scaffold

# FedBC


# python main_fed.py --eval_local --alg fedbc --lr 0.01 --lamb_lr 0.0005 --epochs 100 --skewness_factor 1.2 --classes_per_partition 1 --num_users 20 --local_bs 20 --bs 128 --seed 1 --dataset cifar10 --model cifarCNN 
