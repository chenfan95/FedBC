# FedBC

This is the official implementation of **_FedBC: Calibrating Global and Local Models via
Federated Learning Beyond Consensus_** 

# Requirements
1. Python >= 3.6
2. Pytorch >= 1.8
3. Torchvision >= 0.9

# Dataset
- Synthetic (power law)
- MNIST (power law, iid vs. non-iid)
- CIFAR-10 (power law, iid vs non-iid)
- Shakespeare (adpated from https://github.com/TalwalkarLab/leaf)

# Algorithms 
This repo implements the folloing algorithms
- FedAvg (https://arxiv.org/abs/1602.05629)
- q-FedAvg (https://arxiv.org/abs/1905.10497)
- FedProx (https://arxiv.org/abs/1812.06127)
- Scaffold (https://arxiv.org/abs/1910.06378)
- Per-FedAvg (https://arxiv.org/abs/2002.07948)
- pFedMe (https://arxiv.org/abs/2006.08848)
- **_FedBC_**
- **_Per-FedBC_**

# Argument Parser  
| Argument | Description |
| --- | --- |
| --epochs | number of communication rounds, default = 10|
| --mu | regularization parameter for FedProx c.f (4) in the paper, default = 0|
| --gamma | penalization constant c.f (7) in the paper, default = 0|
| --num_users | total number of users, default = 30|
| --num_local_users | number of users subsampled at each round, default = 10|
| --local_ep | number of local training epochs, default = 5|
| --local_bs | batch size used for local training, default = 20|
| --bs | batch size used for testing, default = 128|
|--alpha| parameter for synthetic dataset, controlling model variations, default = 0.5|
|--beta| parameter for synthetic dataset, controlling data variations, default = 0.5|
|--lr| learning rate for local training, default = 0.01|
|--lamb_lr| learning rate for $\lambda$, default = 0.001|
|--momentum| monmentum for local training, default = 0.5|
|--lamb_momentum|momentum for updating $\lambda$, default = 0.5|
|--q| q for q-FedAvg, default = 0|
|--personal_lamda| $\lambda$ for pFedMe, default = |
|--personal_lr|learning rate for local training for pFedMe|
|--K| number of steps for local training|
|--train_MAML|whether to perform MAML-type training for FedAvg or FedBC|
|--test_MAML|whether to test the model trained with MAML|
|--eval_one_step|whether to take one gradient step to evaluate the global model|
|--inner_lr| learning rate for the lower-level problem of MAML|
|--outer_lr| learning rate for the upper-level problme of MAML|
|--model||
|--dataset||
|--classes_per_partition||
|--skewness_factor||
|--alg||
|--iid||
|--fix_gamma||
|--eval_local||
|||

# Data Partitioning

# How to run
- To run synthetic dataset jobs:
  - see examples in synthetic_jobs.sh 
  - bash synthetic_jobs.sh
- To run CIFAR-10 dataset jobs:
  - see examples in cifar_jobs.sh
  - bash cifar_jobs.sh
- To run MNIST dataset jobs:
  - see examples in mnist_jobs.sh
  - bash mnist_jobs.sh
- To run Shakespeare dataste jobs:
  - go the directory data/shakespeare
  - ./preprocess.sh -s niid --sf 0.1 -k 20 -t sample -tf 0.8 (see https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare for more details); this     will generate the train and test dataset in the directory data/shakespeare/data/train and data/shakespeare/data/test respectively. 
  - specify the train and test dataset name using variables TRAIN_DATA_NAME and TEST_DATA_NAME in function generate_shake() from language_utils.py 
  - bash nlp_jobs.sh
