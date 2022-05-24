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
| --epochs | number of communication rounds |
| --mu | regularization parameter for FedProx c.f (4) in the paper|
| --gamma | penalization constant c.f (7) in the paper|
| --num_users | total number of users |
| --num_local_users | number of users subsampled at each round |
| --local_ep | number of local training epochs |
| --local_bs | batch size used for local training|
| --bs | batch size used for testing |
|--alpha| parameter for synthetic dataset|
|--beta| parameter for synthetic dataset|
|--lr| learning rate for local training|
|--lamb_lr| learning rate for $\lambda$|
|--momentum| monmentum for local training|
|--lamb_momentum|momentum for updating $\lambda$|
|--q| q for q-FedAvg|
|--personal_lamda| $lambda$ for pFedMe|
|--personal_lr||
|--K|
$P(A\|B)$|
|--train_MAML||
|--test_MAML||
|--eval_one_step||
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
