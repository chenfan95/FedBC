#!/bin/sh

# FedAvg
python main_fed.py --eval_local --alg fedavg --epochs 50 --local_ep 1 --lr 0.01 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
# q-FedAvg
python main_fed.py --eval_local --alg qfedavg --epochs 50 --local_ep 1 --lr 0.01 --q 1.0 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
# FedProx
python main_fed.py --eval_local --alg fedprox --epochs 50 --local_ep 1 --lr 0.01 --mu 1.0 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
# Scaffold
python main_fed.py --eval_local --alg scaffold --epochs 50 --local_ep 1 --lr 0.1 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
# FedBC
python main_fed.py --eval_local --alg fedbc --epochs 50 --local_ep 1 --lr 0.01 --lamb_lr 1e-3 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  

# Per-FedAvg
python main_fed.py --eval_local --test_MAML --train_MAML --eval_one_step --alg fedavg --epochs 50 --local_ep 1 --lr 0.01 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
# Per-FedBC
python main_fed.py --eval_local --test_MAML --train_MAML --eval_one_step --alg fedbc --epochs 50 --local_ep 1 --lr 0.01 --lamb_lr 1e-3 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
# pFedMe
python main_fed.py --eval_local --alg pfedme --epochs 50 --local_ep 5 --K 5 --lr 0.01 --personal_lr 0.01 --personal_lamda 1.0 --num_users 30 --dataset synthetic --model logistic --alpha 0.5 --beta 0.5 --results_dir results_synthetic  
