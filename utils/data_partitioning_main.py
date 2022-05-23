from data_partitioning_ops import DataPartitioningUtil
import numpy as np
import random

random.seed(1990)
np.random.seed(seed=1990)

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

	for pidx, pidx_values in partitioned_training_data.items():
		print("Partition ID:{}, Data Distribution: {}".format(pidx, pidx_values))

	return partitioned_training_data

if __name__=="__main__":
	print("k")

	INPUT_DATA = np.array([
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
		[255, 233], [267, 768], [350, 460], [255, 233], [267, 768], [255, 233], [267, 768], [350, 460], [255, 233], [267, 768],
	])

	OUTPUT_CLASSES = np.array([
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	])
	print(INPUT_DATA.shape, OUTPUT_CLASSES.shape)

	#############################################
	### UNIFORM & IID ####
	##############################################
	PARTITIONS_NUM = 10
	CLASSES_PER_PARTITION = 10
	SKEWNESS_FACTOR = 0.0
	BALANCED_RANDOM_PARTITIONING = True
	BALANCED_CLASS_PARTITIONING = False
	UNBALANCED_CLASS_PARTITIONING = False
	STRICTLY_UNBALANCED = False

	print("UNIFORM & IID")
	partition_training_data(partitions_num=PARTITIONS_NUM,
							input_data=INPUT_DATA,
							output_classes=OUTPUT_CLASSES,
							balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
							balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
							unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
							skewness_factor=SKEWNESS_FACTOR,
							strictly_unbalanced=STRICTLY_UNBALANCED,
							classes_per_partition=CLASSES_PER_PARTITION)
	print()

	#############################################
	### UNIFORM & Non-IID(3) ####
	#############################################
	PARTITIONS_NUM = 10
	CLASSES_PER_PARTITION = 3
	SKEWNESS_FACTOR = 0.0
	BALANCED_RANDOM_PARTITIONING = False
	BALANCED_CLASS_PARTITIONING = True
	UNBALANCED_CLASS_PARTITIONING = False
	STRICTLY_UNBALANCED = False

	print("UNIFORM & Non-IID(3)")
	partition_training_data(partitions_num=PARTITIONS_NUM,
							input_data=INPUT_DATA,
							output_classes=OUTPUT_CLASSES,
							balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
							balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
							unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
							skewness_factor=SKEWNESS_FACTOR,
							strictly_unbalanced=STRICTLY_UNBALANCED,
							classes_per_partition=CLASSES_PER_PARTITION)
	print()

	#############################################
	### SKEWED & IID ####
	##############################################
	PARTITIONS_NUM = 20
	CLASSES_PER_PARTITION = 2
	SKEWNESS_FACTOR = 1.2
	BALANCED_RANDOM_PARTITIONING = False
	BALANCED_CLASS_PARTITIONING = False
	UNBALANCED_CLASS_PARTITIONING = True
	STRICTLY_UNBALANCED = False

	print("SKEWED & IID")
	partition_training_data(partitions_num=PARTITIONS_NUM,
							input_data=INPUT_DATA,
							output_classes=OUTPUT_CLASSES,
							balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
							balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
							unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
							skewness_factor=SKEWNESS_FACTOR,
							strictly_unbalanced=STRICTLY_UNBALANCED,
							classes_per_partition=CLASSES_PER_PARTITION)
	print()

	#############################################
	### SKEWED & Non-IID(3) ####
	#############################################
	PARTITIONS_NUM = 10
	CLASSES_PER_PARTITION = 3
	SKEWNESS_FACTOR = 1.5
	BALANCED_RANDOM_PARTITIONING = False
	BALANCED_CLASS_PARTITIONING = False
	UNBALANCED_CLASS_PARTITIONING = True
	STRICTLY_UNBALANCED = False

	print("SKEWED & Non-IID(3)")
	partition_training_data(partitions_num=PARTITIONS_NUM,
							input_data=INPUT_DATA,
							output_classes=OUTPUT_CLASSES,
							balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
							balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
							unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
							skewness_factor=SKEWNESS_FACTOR,
							strictly_unbalanced=STRICTLY_UNBALANCED,
							classes_per_partition=CLASSES_PER_PARTITION)
	print()

	#############################################
	### POWER LAW & IID ####
	##############################################
	PARTITIONS_NUM = 10
	CLASSES_PER_PARTITION = 10
	SKEWNESS_FACTOR = 1.5
	BALANCED_RANDOM_PARTITIONING = False
	BALANCED_CLASS_PARTITIONING = False
	UNBALANCED_CLASS_PARTITIONING = True
	STRICTLY_UNBALANCED = True

	print("POWER LAW & IID")
	partition_training_data(partitions_num=PARTITIONS_NUM,
							input_data=INPUT_DATA,
							output_classes=OUTPUT_CLASSES,
							balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
							balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
							unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
							skewness_factor=SKEWNESS_FACTOR,
							strictly_unbalanced=STRICTLY_UNBALANCED,
							classes_per_partition=CLASSES_PER_PARTITION)
	print()

	#############################################
	### POWER LAW & Non-IID(3) ####
	#############################################
	PARTITIONS_NUM = 10
	CLASSES_PER_PARTITION = 3
	SKEWNESS_FACTOR = 1.5
	BALANCED_RANDOM_PARTITIONING = False
	BALANCED_CLASS_PARTITIONING = False
	UNBALANCED_CLASS_PARTITIONING = True
	STRICTLY_UNBALANCED = True

	print("POWER LAW & Non-IID(3)")
	partition_training_data(partitions_num=PARTITIONS_NUM,
							input_data=INPUT_DATA,
							output_classes=OUTPUT_CLASSES,
							balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
							balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
							unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
							skewness_factor=SKEWNESS_FACTOR,
							strictly_unbalanced=STRICTLY_UNBALANCED,
							classes_per_partition=CLASSES_PER_PARTITION)
	print()



