# Import libraries and modules
import csv
import os
from dataset import paths as ds
import numpy as np

"""
This module contains all the functions to manage dataset files, folders, sequences and efficiency values
"""


def extract_all_sets(dataset=ds.default_dataset_path):
    """
    Extracts the 10 main datasets from Haeussler dataset file

    :param dataset: path of the Haeussler dataset file
    :return: None
    """

    # Create training sets directory if it does not exist
    if not os.path.isdir(ds.training_set_folder):
        os.mkdir(ds.training_set_folder)

    # If chari_293t, wangxu_hl60 and doench_mel4 datasets are not extracted from Haeussler, extract them
    if not os.path.isfile(ds.training_set_folder + ds.chari_293t) or not os.path.isfile(
            ds.training_set_folder + ds.wangxu_hl60) or not os.path.isfile(ds.training_set_folder + ds.doench_mel4):

        # Create dataset files and writers
        chari_train_file = open(ds.training_set_folder + ds.chari_293t, 'w')
        chari_train_writer = csv.writer(chari_train_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        wangxu_train_file = open(ds.training_set_folder + ds.wangxu_hl60, 'w')
        wangxu_train_writer = csv.writer(wangxu_train_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        doench_train_file = open(ds.training_set_folder + ds.doench_mel4, 'w')
        doench_train_writer = csv.writer(doench_train_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Open Haeussler dataset file and reader
        tsv_dataset_file = open(dataset, 'r')
        tsv_reader = csv.reader(tsv_dataset_file, delimiter='\t')

        # For each line in the dataset
        for line in tsv_reader:

            # If current line belongs to chari_293t dataset
            if line[0] == ds.chari_293t_id:

                # Extract the 30-bp sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency in chari_293T dataset file
                chari_train_writer.writerow([extracted_sequence, line[3]])

            # If current line belongs to wangxu_hl60 dataset
            elif line[0] == ds.wangxu_hl60_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to wangxu_hl60 dataset file
                wangxu_train_writer.writerow([extracted_sequence, line[3]])

            # If current line belongs to doench_mel4 dataset
            elif line[0] == ds.doench_mel4_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to doench_mel4 dataset file
                doench_train_writer.writerow([extracted_sequence, line[3]])

        # Close all files
        tsv_dataset_file.close()
        chari_train_file.close()
        wangxu_train_file.close()
        doench_train_file.close()

    # If doench_hg19, hart_hct116, moreno_mateos, ghandi_ci2, farboud, varshney and gagnon datasets are not extracted
    # from Haeussler dataset, extract them
    if not os.path.isfile(ds.training_set_folder + ds.doench_hg19) or not os.path.isfile(
            ds.training_set_folder + ds.hart_hct116) or not os.path.isfile(
        ds.training_set_folder + ds.moreno_mateos) or not os.path.isfile(
        ds.training_set_folder + ds.gandhi_ci2) or not os.path.isfile(
        ds.training_set_folder + ds.farboud) or not os.path.isfile(
        ds.training_set_folder + ds.varshney) or not os.path.isfile(ds.training_set_folder + ds.gagnon):

        # Create dataset files and writers
        doench_test_file = open(ds.training_set_folder + ds.doench_hg19, 'w')
        doench_test_writer = csv.writer(doench_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hart_test_file = open(ds.training_set_folder + ds.hart_hct116, 'w')
        hart_test_writer = csv.writer(hart_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        moreno_mateos_test_file = open(ds.training_set_folder + ds.moreno_mateos, 'w')
        moreno_mateos_test_writer = csv.writer(moreno_mateos_test_file, delimiter='\t', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
        ghandi_test_file = open(ds.training_set_folder + ds.gandhi_ci2, 'w')
        ghandi_test_writer = csv.writer(ghandi_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        farboud_test_file = open(ds.training_set_folder + ds.farboud, 'w')
        farboud_test_writer = csv.writer(farboud_test_file, delimiter='\t', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
        varshney_test_file = open(ds.training_set_folder + ds.varshney, 'w')
        varshney_test_writer = csv.writer(varshney_test_file, delimiter='\t', quotechar='"',
                                          quoting=csv.QUOTE_MINIMAL)
        gagnon_test_file = open(ds.training_set_folder + ds.gagnon, 'w')
        gagnon_test_writer = csv.writer(gagnon_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Open Haeussler file and reader
        tsv_dataset_file = open(dataset, 'r')
        tsv_reader = csv.reader(tsv_dataset_file, delimiter='\t')

        # For each line in the dataset
        for line in tsv_reader:

            # If the current line belongs to doench_hg19 dataset
            if line[0] == ds.doench_hg19_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to doench_hg19 dataset file
                doench_test_writer.writerow([extracted_sequence, line[3]])

            # If the current line belongs to hart_hct116 dataset
            elif line[0] == ds.hart_hct116_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to hart_hct116 dataset file
                hart_test_writer.writerow([extracted_sequence, line[3]])

            # If the current line belongs to moreno_mateos dataset
            elif line[0] == ds.moreno_mateos_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to moreno_mateos dataset file
                moreno_mateos_test_writer.writerow([extracted_sequence, line[3]])

            # If the current line belongs to gandhi_ci2 dataset
            elif line[0] == ds.gandhi_ci2_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to gandhi_ci2 dataset file
                ghandi_test_writer.writerow([extracted_sequence, line[3]])

            # If the current line belongs to farboud dataset
            elif line[0] == ds.farboud_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to farboud dataset file
                farboud_test_writer.writerow([extracted_sequence, line[3]])

            # If the current line belongs to varshney dataset
            elif line[0] == ds.varshney_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to varshney dataset file
                varshney_test_writer.writerow([extracted_sequence, line[3]])

            # If the current line belongs to gagnon dataset
            elif line[0] == ds.gagnon_id:

                # Extract the 30-bp full sequence from longSeq column
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]

                # Save 30-bp sequence and its relative efficiency to gagnon dataset file
                gagnon_test_writer.writerow([extracted_sequence, line[3]])

        # Close all files
        tsv_dataset_file.close()
        gagnon_test_file.close()
        varshney_test_file.close()
        farboud_test_file.close()
        ghandi_test_file.close()
        moreno_mateos_test_file.close()
        hart_test_file.close()
        doench_test_file.close()


def get_min_max_efficiency(dataset):
    """
    Extract the minimum and maximum efficiency value from the given dataset

    :param dataset: path of the dataset to extract min and max from
    :return: minimum efficiency and maximum efficiency of the dataset
    """

    # Open given dataset file and reader
    csv_dataset_file = open(dataset, 'r')
    csv_reader = csv.reader(csv_dataset_file, delimiter='\t')

    col = []  # Efficiency list representing the efficiency column on the dataset file

    # For each row in the file
    for row in csv_reader:
        # Append the current efficiency value to the efficiency list
        col.append(float(row[1]))

    # Close file
    csv_dataset_file.close()

    return min(col), max(col)


def rescale(dataset):
    """
    Rescale given dataset using its minimum and maximum efficiency

    :param dataset: path of the dataset to rescale
    :return: None
    """

    # Open given dataset file and reader
    dataset_file = open(dataset, 'r')
    dataser_reader = csv.reader(dataset_file, delimiter='\t')

    # Get the minimum and maximum efficiency of the given dataset
    min_efficiency, max_efficiency = get_min_max_efficiency(dataset)

    # Create rescaled dataset file and writer
    rescaled_dataset_file = open(ds.rescaled_train_set_folder + os.path.basename(dataset), 'w')
    rescaled_dataset_writer = csv.writer(rescaled_dataset_file, delimiter='\t', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)

    # For each row in the given dataset
    for row in dataser_reader:
        # Rescale the current efficiency using minmax rescaler
        rescaled_efficiency = minmax_rescaler(min_efficiency, max_efficiency, float(row[1]))

        # Write the current sequence and its rescaled efficiency to rescaled dataset
        rescaled_dataset_writer.writerow([row[0], rescaled_efficiency])

    # Close all files
    dataset_file.close()
    rescaled_dataset_file.close()


def minmax_rescaler(min_efficiency, max_efficiency, efficiency):
    """
    Rescale a value of efficiency in [min_efficiency, max_efficiency] range to a value in range [0,1]

    :param min_efficiency: minimum value of efficiency
    :param max_efficiency: maximum value of efficiency
    :param efficiency: efficiency value to rescale
    :return: rescaled efficiency value in the range [0,1]
    """
    return (efficiency - min_efficiency) / (max_efficiency - min_efficiency)


def rescale_all_sets():
    """
    Rescale all efficiencies of the sets in the training set directory

    :return: None
    """

    # Create rescaled sets folder if it does not exist
    if not os.path.isdir(ds.rescaled_set_folder):
        os.mkdir(ds.rescaled_set_folder)

    # Create rescaled training sets folder if it does not exist
    if not os.path.isdir(ds.rescaled_train_set_folder):
        os.mkdir(ds.rescaled_train_set_folder)

    # For each dataset in training sets folder
    for (_, _, filenames) in os.walk(ds.training_set_folder):
        for elem in filenames:
            file = ds.training_set_folder + str(elem)
            folder = "dataset/"
            if file.startswith(folder):
                file = file[len(folder):]

            # Rescale current dataset if not already rescaled
            if not os.path.isfile(ds.rescaled_set_folder + file):
                rescale(folder + file)


def encode_sgrna_sequence(sequence):
    """
    Encode a sgRNA sequence into a one-hot matrix

    :param sequence: string representing the sgRNA sequence to encode
    :return: one-hot encoded matrix
    """

    # If the given sequence is longer than 30 or shorter than 0 throw and exception
    if len(sequence) > 30:
        raise Exception("Sequence is too long")
    elif len(sequence) <= 0:
        raise Exception("Sequence error (registered a sequence with negative or zero length)")

    # Else encode the sequence into a 4x30 one-hot matrix; if the sequence is shorter than 30, add 30 - len(sequence)
    # columns of zeros at the beginning of the matrix
    else:
        one_hot_matrix = np.zeros((4, 30))

        for i in range(len(sequence)):
            if sequence[i] == 'A':
                one_hot_matrix[0, i + 30 - len(sequence)] = 1
            elif sequence[i] == 'C':
                one_hot_matrix[1, i + 30 - len(sequence)] = 1
            elif sequence[i] == 'G':
                one_hot_matrix[2, i + 30 - len(sequence)] = 1
            elif sequence[i] == 'T':
                one_hot_matrix[3, i + 30 - len(sequence)] = 1
            else:
                raise Exception("Dataset contains an uncorrect sgRNA sequence: " + sequence)

        return one_hot_matrix


def encode(dataset):
    """
    Encode the given dataset sequences

    :param dataset: path of the dataset file to encode
    :return: a list of encoded sequences and a list of their respective efficiencies
    """

    # Open the given dataset file
    dataset_file = open(dataset, 'r')
    dataser_reader = csv.reader(dataset_file, delimiter='\t')

    # Create an empty list of encoded sequences and an empty list of their respective efficiencies
    sequence_array = []
    efficiency_array = []

    # For each row in the dataset file
    for row in dataser_reader:
        # Encode the current sequence and add it to the encoded sequences list
        sequence_array.append(encode_sgrna_sequence(str(row[0])))

        # Add the current sequence efficiency to the efficiency list
        efficiency_array.append(float(row[1]))

    # Close dataset file
    dataset_file.close()

    return sequence_array, efficiency_array


def encode_all_augmented_sets():
    """
    Encode all sequences of all sets in the augmented training sets folder

    :return: a list of encoded datasets
    """

    # Create an empty list of encoded datasets
    datasets_array = []

    # For each dataset in the augmented training sets folder
    for (_, _, filenames) in os.walk(ds.augmented_train_set_folder):
        for elem in filenames:
            file = ds.augmented_train_set_folder + str(elem)

            # Encode the current dataset
            sequence_array, efficiency_array = encode(file)

            # Add the encoded dataset and its name to the encoded dataset list
            datasets_array.append([sequence_array, efficiency_array, os.path.splitext(os.path.basename(file))[0]])

    return datasets_array


def encode_all_sets():
    """
    Encode all sequences of all sets in the rescaled training sets folder

    :return: a list of encoded datasets
    """

    # Create an empty list of encoded datasets
    datasets_array = []

    # For each dataset in the augmented training sets folder
    for (_, _, filenames) in os.walk(ds.rescaled_train_set_folder):
        for elem in filenames:
            file = ds.rescaled_train_set_folder + str(elem)

            # Encode the current dataset
            sequence_array, efficiency_array = encode(file)

            # Add the encoded dataset and its name to the encoded dataset list
            datasets_array.append([sequence_array, efficiency_array, os.path.splitext(os.path.basename(file))[0]])

    return datasets_array


def augment_all_sets():
    """
    Augment all sets in the rescaled training sets folder

    :return: None
    """

    # Create an augmented sets folder if it does not exist
    if not os.path.isdir(ds.augmented_set_folder):
        os.mkdir(ds.augmented_set_folder)

    # Create an augmented training sets folder if it does not exist
    if not os.path.isdir(ds.augmented_train_set_folder):
        os.mkdir(ds.augmented_train_set_folder)

    # For each dataset in the rescaled training sets folder
    for (_, _, filenames) in os.walk(ds.rescaled_train_set_folder):
        for elem in filenames:
            file = ds.rescaled_train_set_folder + str(elem)

            # Augment the current dataset
            augment(file)


def augment(dataset):
    """
    Augment the given dataset sequences generating permutations in the first two position of the PAM-distal region
    (5' end)

    :param dataset: path of the dataset file to augment
    :return: None
    """

    # Open given dataset file and reader
    dataset_file = open(dataset, 'r')
    dataser_reader = csv.reader(dataset_file, delimiter='\t')

    augmented_file_name = ds.augmented_train_set_folder + str(os.path.basename(dataset))

    # Create the augmented dataset file if it does not exist
    if not os.path.isfile(augmented_file_name):
        augmented_dataset_file = open(augmented_file_name, 'w')
        augmented_dataset_writer = csv.writer(augmented_dataset_file, delimiter='\t', quotechar='"',
                                              quoting=csv.QUOTE_MINIMAL)

        # For each row in the given dataset
        for row in dataser_reader:

            # Generate a list of augmented sequences of the current sequence
            augmented_sequences = augment_sgrna_sequence(str(row[0]))

            # For each sequence in the augmented sequences list
            for sequence in augmented_sequences:
                # Write the current augmented sequence with the original sequence efficiency to the augmented set file
                augmented_dataset_writer.writerow([sequence, float(row[1])])

        # Close augmented set file
        augmented_dataset_file.close()

    # Close given dataset file
    dataset_file.close()


def augment_sgrna_sequence(sequence):
    """
    Generate permutations of the given sequence in the first two position of the PAM-distal region (5' end)

    :param sequence: the sequence to augment
    :return: a list containing the augmented sequences
    """

    # Throw an exception if sequence is longer than 30 or shorter/equal than 0
    if len(sequence) > 30:
        raise Exception("Sequence is too long")
    elif len(sequence) <= 0:
        raise Exception("Sequence error (registered a sequence with negative or zero length)")

    # Else augment the given sequence creating permutations
    else:
        augmented_sequence_list = []
        nucleobasis = ["A", "C", "G", "T"]
        for base1 in nucleobasis:
            for base2 in nucleobasis:
                augmented_sequence_list.append(sequence[0:7] + base1 + base2 + sequence[9:])

        return augmented_sequence_list
