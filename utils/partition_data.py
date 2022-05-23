from utils.data_partitioning_ops import DataPartitioningUtil
#from data_partitioning_ops import DataPartitioningUtil
import numpy as np
import random
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.options import args_parser
#from options import args_parser
import pickle

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(seed=0)

def partition_training_data(input_data, output_classes, partitions_num, balanced_random_partitioning=False,
							balanced_class_partitioning=False, classes_per_partition=None,
							unbalanced_class_partitioning=False, skewness_factor=None, strictly_unbalanced=None):
	partitioned_training_data = None
	if balanced_random_partitioning:
		partition_policy = "IID"
		partitioned_training_data = DataPartitioningUtil.balanced_random_partitioning(
			input_data=input_data,
			output_classes=output_classes,
			partitions=partitions_num)

	elif balanced_class_partitioning:
		if classes_per_partition is None:
			raise RuntimeError("`classes_per_partition` must be defined when class partitioning is invoked")
		partitioned_training_data = DataPartitioningUtil.balanced_class_partitioning(
			input_data=input_data,
			output_classes=output_classes,
			partitions=partitions_num,
			classes_per_partition=classes_per_partition)

	elif unbalanced_class_partitioning is True and skewness_factor is not None:

		if skewness_factor is None:
			raise RuntimeError("You need to specify the skewness factor of the distribution.")
		if strictly_unbalanced is None:
			raise RuntimeError("You need to specify if you want to create a strictly unbalanced distribution.")

		if strictly_unbalanced is False:
			partitioned_training_data = DataPartitioningUtil.strictly_noniid_unbalanced_data_partitioning(
				input_data=input_data,
				output_classes=output_classes,
				partitions=partitions_num,
				classes_per_partition=classes_per_partition,
				skewness_factor=skewness_factor)
		else:
			partitioned_training_data = DataPartitioningUtil.strictly_unbalanced_noniid_data_partitioning(
				input_data=input_data,
				output_classes=output_classes,
				partitions=partitions_num,
				classes_per_partition=classes_per_partition,
				skewness_factor=skewness_factor)

	else:
		raise RuntimeError("You need to specify at least one data partitioning scheme")

	partition_ids_by_descending_partition_size = sorted(
		partitioned_training_data,
		key=lambda partition_id: partitioned_training_data[partition_id].partition_size, reverse=True)
	new_sorted_session_partitioned_data = dict()
	for new_pidx, partition_id in enumerate(partition_ids_by_descending_partition_size):
		new_sorted_session_partitioned_data[new_pidx] = partitioned_training_data[partition_id]

	partitioned_training_data = new_sorted_session_partitioned_data

	# for pidx, pidx_values in partitioned_training_data.items():
	# 	print("Partition ID:{}, Data Distribution: {}".format(pidx, pidx_values))

	return partitioned_training_data

class img_dataset(Dataset):

    def __init__(self, data, target, transform=None, target_transform=None):

        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

transform_train_cifar = transforms.Compose([
    #transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_cifar = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_mnist =transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train_test_split = 0.8

def generate_power_dataset(args):
    # return the dataloader for the entire train/test dataset
    # as well as the dataloader for each user's train/test dataset
    
    #cifar_dataset_train = img_dataset(TRAIN_DATA.data, TRAIN_DATA.targets, transform=transform_train_cifar)
    #cifar_dataset_test = img_dataset(TEST_DATA.data, TEST_DATA.targets, transform=transform_test_cifar)

    partitioned_data, partition_stats = partition_data_driver(args)
    trainDataset_local = {}
    testDataset_local = {}    
    for pidx, pidx_values in partitioned_data.items():
        #print("Partition ID:{}, Data Distribution: {} {}".format(pidx, pidx_values.partition_classes, pidx_values.partition_size))
        
        X = pidx_values.input
        y = pidx_values.output

        num_samples = len(X)
        train_len = int(train_test_split * num_samples)
        test_len = num_samples - train_len
        
        X_train_i = X[:train_len]
        X_test_i = X[train_len:]
        y_train_i = y[:train_len]
        y_test_i = y[train_len:]
        
        if pidx == 0:
            trainData_X = X_train_i
            testData_X = X_test_i
            trainData_y = y_train_i
            testData_y = y_test_i
        else:
            trainData_X = np.concatenate((trainData_X, X_train_i))
            testData_X = np.concatenate((testData_X, X_test_i))
            trainData_y = np.concatenate((trainData_y, y_train_i))
            testData_y = np.concatenate((testData_y, y_test_i))

        if args.dataset == "cifar10":
            trainDataset_local[pidx] = img_dataset(X_train_i, y_train_i, transform = transform_train_cifar) 
            testDataset_local[pidx] = img_dataset(X_test_i, y_test_i, transform = transform_test_cifar) 
            trainDataset = img_dataset(trainData_X, trainData_y, transform = transform_train_cifar)
            testDataset = img_dataset(testData_X, testData_y, transform = transform_test_cifar)
        
        elif args.dataset == "mnist":
            trainDataset_local[pidx] = img_dataset(X_train_i, y_train_i, transform = transform_mnist) 
            testDataset_local[pidx] = img_dataset(X_test_i, y_test_i, transform = transform_mnist) 
            trainDataset = img_dataset(trainData_X, trainData_y, transform = transform_mnist)
            testDataset = img_dataset(testData_X, testData_y,transform = transform_mnist)
    
    return trainDataset, testDataset, trainDataset_local, testDataset_local, partition_stats

def partition_data_driver(args):

	#############################################
	### SKEWED & Non-IID(3) ####
	#############################################
	
    # PARTITIONS_NUM = 30
    # CLASSES_PER_PARTITION = 1
    # SKEWNESS_FACTOR = 1.2
    # BALANCED_RANDOM_PARTITIONING = False
    # BALANCED_CLASS_PARTITIONING = False
    # UNBALANCED_CLASS_PARTITIONING = True
    # STRICTLY_UNBALANCED = False

    # print("SKEWED & Non-IID(3)")
    # partitioned_data = partition_training_data(partitions_num=PARTITIONS_NUM,
    #                         input_data=INPUT_DATA,
    #                         output_classes=OUTPUT_CLASSES,
    #                         balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
    #                         balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
    #                         unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
    #                         skewness_factor=SKEWNESS_FACTOR,
    #                         strictly_unbalanced=STRICTLY_UNBALANCED,
    #                         classes_per_partition=CLASSES_PER_PARTITION)


    ############################################
    ## POWER LAW & Non-IID(3) ####
    ############################################

    if args.dataset == "cifar10": 
        TRAIN_DATA = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
        TEST_DATA = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True)
        INPUT_DATA = np.concatenate((TRAIN_DATA.data, TEST_DATA.data)) 
        OUTPUT_CLASSES = np.array(TRAIN_DATA.targets+TEST_DATA.targets)
    
    elif args.dataset == "mnist":
        TRAIN_DATA = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True)
        TEST_DATA = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True)
        INPUT_DATA = np.concatenate((TRAIN_DATA.data, TEST_DATA.data)) 
        OUTPUT_CLASSES = np.concatenate((TRAIN_DATA.targets.numpy(), TEST_DATA.targets.numpy()))
    
    #OUTPUT_CLASSES = np.array(TRAIN_DATA.targets + TEST_DATA.targets)
    print(INPUT_DATA.shape)
    print(OUTPUT_CLASSES.shape)


	#############################################
	### SKEWED & IID ####
	##############################################
	# PARTITIONS_NUM = 20
	# CLASSES_PER_PARTITION = 2
	# SKEWNESS_FACTOR = 1.2
	# BALANCED_RANDOM_PARTITIONING = False
	# BALANCED_CLASS_PARTITIONING = False
	# UNBALANCED_CLASS_PARTITIONING = True
	# STRICTLY_UNBALANCED = False

    if args.iid == True:
        # PARTITIONS_NUM = args.num_users
        # CLASSES_PER_PARTITION = 10
        # SKEWNESS_FACTOR = args.skewness_factor
        # BALANCED_RANDOM_PARTITIONING = False
        # BALANCED_CLASS_PARTITIONING = False
        # UNBALANCED_CLASS_PARTITIONING = True
        # STRICTLY_UNBALANCED = False

        PARTITIONS_NUM = args.num_users
        CLASSES_PER_PARTITION = 10
        SKEWNESS_FACTOR = 0.0
        BALANCED_RANDOM_PARTITIONING = False
        BALANCED_CLASS_PARTITIONING = True
        UNBALANCED_CLASS_PARTITIONING = False
        STRICTLY_UNBALANCED = False

    else:
        PARTITIONS_NUM = args.num_users
        CLASSES_PER_PARTITION = args.classes_per_partition
        SKEWNESS_FACTOR = args.skewness_factor
        BALANCED_RANDOM_PARTITIONING = False
        BALANCED_CLASS_PARTITIONING = False
        UNBALANCED_CLASS_PARTITIONING = True
        STRICTLY_UNBALANCED = True
    
    #print("POWER LAW & Non-IID(3)")
    partitioned_data = partition_training_data(partitions_num=PARTITIONS_NUM,
                            input_data=INPUT_DATA,
                            output_classes=OUTPUT_CLASSES,
                            balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
                            balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
                            unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
                            skewness_factor=SKEWNESS_FACTOR,
                            strictly_unbalanced=STRICTLY_UNBALANCED,
                            classes_per_partition=CLASSES_PER_PARTITION)
    #print(partitioned_data.keys())
    #print(partitioned_data.items())

    partition_stats = {}
    for pidx, pidx_values in partitioned_data.items():
        print("Partition ID:{}, Data Distribution: {} {}".format(pidx, pidx_values.partition_classes, pidx_values.partition_size))
        partition_stats[pidx] = (pidx_values.partition_classes, pidx_values.partition_size)
    #print(partitioned_data[0].__dict__)
    return partitioned_data, partition_stats

# if __name__=="__main__":
 
#     args = args_parser()
#     args.dataset = "cifar10" 
#     args.skewness_factor = 1.5
#     args.classes_per_partition = 3
#     args.num_users = 10
#     partitioned_data, partition_stats = partition_data_driver(args)
    #output_file = "../Results/cifar10_partition_R100_test" + ".p"
    #pickle.dump(partition_stats, open(output_file, "wb" ))
        
    #trainDataset, testDataset, trainDataset_local, testDataset_local, partition_stats = generate_power_dataset(args)
    


