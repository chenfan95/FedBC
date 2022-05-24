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
