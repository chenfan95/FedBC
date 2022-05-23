#!/bin/sh

# num_users = 20
python partition.py --alg fedavg --seed 1 --skewness_factor 1.1 --classes_per_partition 1 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.2 --classes_per_partition 1 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.3 --classes_per_partition 1 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.4 --classes_per_partition 1 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.5 --classes_per_partition 1 --num_users 20 --dataset cifar10 --model cifarCNN

python partition.py --alg fedavg --seed 1 --skewness_factor 1.1 --classes_per_partition 2 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.2 --classes_per_partition 2 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.3 --classes_per_partition 2 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.4 --classes_per_partition 2 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.5 --classes_per_partition 2 --num_users 20 --dataset cifar10 --model cifarCNN

python partition.py --alg fedavg --seed 1 --skewness_factor 1.1 --classes_per_partition 3 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.2 --classes_per_partition 3 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.3 --classes_per_partition 3 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.4 --classes_per_partition 3 --num_users 20 --dataset cifar10 --model cifarCNN
python partition.py --alg fedavg --seed 1 --skewness_factor 1.5 --classes_per_partition 3 --num_users 20 --dataset cifar10 --model cifarCNN
