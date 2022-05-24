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
|--personal_lamda| $\lambda$ for pFedMe, default = 0.01|
|--personal_lr|learning rate for local training for pFedMe, default = 0.01|
|--K| number of steps for local training, default = 1|
|--train_MAML|whether to perform MAML-type training for FedAvg or FedBC, default = False|
|--test_MAML|whether to test the model trained with MAML, default =  False|
|--eval_one_step|whether to take one gradient step to evaluate the global model, default = False|
|--inner_lr| learning rate for the lower-level problem of MAML, default = 0.01|
|--outer_lr| learning rate for the upper-level problme of MAML, default = 0.001|
|--model| models for different datasets, choices: cifarCNN, logistic, mlp, rnn, default = cifarCNN|
|--dataset| choices: mnist, cifar10, sythetic, shakespeare, default = mnist|
|--classes_per_partition| the most common number of classes own by users, default = 2|
|--skewness_factor| power law distribution exponent, default = 0.0|
|--alg| choices: fedavg, qfedavg, fedprox, scaffold, fedbc,pfedme, default = fedavg|
|--iid| whether the user's local data is iid or not for mnist, cifar10 and synthetic|
|--fix_gamma| whether $\gamma$ is fixed for , default = False|
|--eval_local| whether to evaluate local performance, default = False|
|--gpu| whether to use gpu or not, default = 0|
|--seed| random seed, default = 1|
|--results_dir| directory for results, default = Results|
|--partition_dir| directory for data partitions, default = Partitions|
|--read_partition| whether to read data partition, default = False|

# Data Partitioning
- The two important parameters are "skewness_factor" (power law exponent) and 
and "classes_per_partition", which control data size heterogeneity and non-iidness respectively. The higher 
the skewness factor and the fewer classes per partition, the more heterogeneous the data becomes
- The function for creating data partitions is "partition.py". To create partitions for MNIST, CIFAR-10, synthetic:
  - bash create_partition_cifar.sh
  - bash create_partition_mnist.sh
  - bash create_partition_synthetic.sh

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
