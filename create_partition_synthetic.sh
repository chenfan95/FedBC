#!/bin/sh

python partition.py --num_users 30 --seed 1 --dataset synthetic 
python partition.py --num_users 30 --seed 1 --dataset synthetic --iid
