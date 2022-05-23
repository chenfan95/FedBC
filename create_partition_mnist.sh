#!/bin/sh

python partition.py --alg fedavg --seed 1 --skewness_factor 1.1 --classes_per_partition 1 --num_users 20 --dataset mnist --model logistic 
python partition.py --alg fedavg --seed 1 --skewness_factor 1.1 --classes_per_partition 2 --num_users 20 --dataset mnist --model logistic
python partition.py --alg fedavg --seed 1 --skewness_factor 1.1 --classes_per_partition 3 --num_users 30 --dataset mnist --model logistic

python partition.py --alg fedavg --seed 1 --skewness_factor 1.2 --classes_per_partition 1 --num_users 30 --dataset mnist --model logistic
python partition.py --alg fedavg --seed 1 --skewness_factor 1.2 --classes_per_partition 2 --num_users 30 --dataset mnist --model logistic
python partition.py --alg fedavg --seed 1 --skewness_factor 1.2 --classes_per_partition 3 --num_users 30 --dataset mnist --model logistic

python partition.py --alg fedavg --seed 1 --skewness_factor 1.3 --classes_per_partition 1 --num_users 30 --dataset mnist --model logistic
python partition.py --alg fedavg --seed 1 --skewness_factor 1.3 --classes_per_partition 2 --num_users 30 --dataset mnist --model logistic
python partition.py --alg fedavg --seed 1 --skewness_factor 1.3 --classes_per_partition 3 --num_users 30 --dataset mnist --model logistic
