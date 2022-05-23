# FedBC

This is the official implementation of **_FedBC: Calibrating Global and Local Models via
Federated Learning Beyond Consensus_**

# Requirements
1. Python >= 3.6
2. Pytorch >= 1.8
3. Torchvision >= 0.9

# Dataset
- Synthetic
- MNIST 
- CIFAR-10 
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
