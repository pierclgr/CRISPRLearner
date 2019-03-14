import csv
import os
from dataset import paths as ds
import numpy as np


def extract_train_sets(dataset=ds.default_dataset_path):
    if not os.path.isdir(ds.training_set_folder):
        os.mkdir(ds.training_set_folder)
    else:
        print("Training sets' folder already exists")

    if not os.path.isfile(ds.training_set_folder + ds.chari_293t) or not os.path.isfile(
            ds.training_set_folder + ds.wangxu_hl60) or not os.path.isfile(ds.training_set_folder + ds.doench_mel4):
        chari_train_file = open(ds.training_set_folder + ds.chari_293t, 'w')
        chari_train_writer = csv.writer(chari_train_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        wangxu_train_file = open(ds.training_set_folder + ds.wangxu_hl60, 'w')
        wangxu_train_writer = csv.writer(wangxu_train_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        doench_train_file = open(ds.training_set_folder + ds.doench_mel4, 'w')
        doench_train_writer = csv.writer(doench_train_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        tsv_dataset_file = open(dataset, 'r')
        tsv_reader = csv.reader(tsv_dataset_file, delimiter='\t')
        for line in tsv_reader:
            if line[0] == ds.chari_293t_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                chari_train_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.wangxu_hl60_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                wangxu_train_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.doench_mel4_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                doench_train_writer.writerow([extracted_sequence, line[3]])

        tsv_dataset_file.close()
        chari_train_file.close()
        wangxu_train_file.close()
        doench_train_file.close()

    else:
        print("Training sets already extracted")


def extract_test_sets(dataset=ds.default_dataset_path):
    if not os.path.isdir(ds.testing_set_folder):
        os.mkdir(ds.testing_set_folder)
    else:
        print("Testing sets' folder already exists")

    if not os.path.isfile(ds.testing_set_folder + ds.doench_hg19) or not os.path.isfile(
            ds.testing_set_folder + ds.hart_hct116) or not os.path.isfile(
        ds.testing_set_folder + ds.moreno_mateos) or not os.path.isfile(
        ds.testing_set_folder + ds.gandhi_ci2) or not os.path.isfile(
        ds.testing_set_folder + ds.farboud) or not os.path.isfile(
        ds.testing_set_folder + ds.varshney) or not os.path.isfile(ds.testing_set_folder + ds.gagnon):
        doench_test_file = open(ds.testing_set_folder + ds.doench_hg19, 'w')
        doench_test_writer = csv.writer(doench_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hart_test_file = open(ds.testing_set_folder + ds.hart_hct116, 'w')
        hart_test_writer = csv.writer(hart_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        moreno_mateos_test_file = open(ds.testing_set_folder + ds.moreno_mateos, 'w')
        moreno_mateos_test_writer = csv.writer(moreno_mateos_test_file, delimiter='\t', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
        ghandi_test_file = open(ds.testing_set_folder + ds.gandhi_ci2, 'w')
        ghandi_test_writer = csv.writer(ghandi_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        farboud_test_file = open(ds.testing_set_folder + ds.farboud, 'w')
        farboud_test_writer = csv.writer(farboud_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        varshney_test_file = open(ds.testing_set_folder + ds.varshney, 'w')
        varshney_test_writer = csv.writer(varshney_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        gagnon_test_file = open(ds.testing_set_folder + ds.gagnon, 'w')
        gagnon_test_writer = csv.writer(gagnon_test_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        tsv_dataset_file = open(dataset, 'r')
        tsv_reader = csv.reader(tsv_dataset_file, delimiter='\t')
        for line in tsv_reader:
            if line[0] == ds.doench_hg19_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                doench_test_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.hart_hct116_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                hart_test_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.moreno_mateos_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                moreno_mateos_test_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.gandhi_ci2_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                ghandi_test_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.farboud_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                farboud_test_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.varshney_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                varshney_test_writer.writerow([extracted_sequence, line[3]])
            elif line[0] == ds.gagnon_id:
                long_seq = str(line[6])
                seq = str(line[2])
                index = long_seq.find(seq)
                start_index = index - (30 - len(seq))
                extracted_sequence = line[6][start_index:start_index + 30]
                gagnon_test_writer.writerow([extracted_sequence, line[3]])

        tsv_dataset_file.close()
        gagnon_test_file.close()
        varshney_test_file.close()
        farboud_test_file.close()
        ghandi_test_file.close()
        moreno_mateos_test_file.close()
        hart_test_file.close()
        doench_test_file.close()

    else:
        print("Test sets already extracted")


def get_min_max_efficiency(dataset):
    csv_dataset_file = open(dataset, 'r')
    csv_reader = csv.reader(csv_dataset_file, delimiter='\t')

    col = []

    for row in csv_reader:
        col.append(float(row[1]))

    csv_dataset_file.close()

    return min(col), max(col)


def rescale(dataset, rescaled_dataset_path):
    dataset_file = open(dataset, 'r')
    dataser_reader = csv.reader(dataset_file, delimiter='\t')

    min_efficiency, max_efficiency = get_min_max_efficiency(dataset)

    rescaled_dataset_file = open(rescaled_dataset_path, 'w')
    rescaled_dataset_writer = csv.writer(rescaled_dataset_file, delimiter='\t', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)

    for row in dataser_reader:
        rescaled_efficiency = minmax_rescaler(min_efficiency, max_efficiency, float(row[1]))
        rescaled_dataset_writer.writerow([row[0], rescaled_efficiency])

    dataset_file.close()
    rescaled_dataset_file.close()


def minmax_rescaler(min_efficiency, max_efficiency, efficiency):
    return (efficiency - min_efficiency) / (max_efficiency - min_efficiency)


def rescale_all_sets():
    if not os.path.isdir(ds.rescaled_set_folder):
        os.mkdir(ds.rescaled_set_folder)
    else:
        print("Rescaled sets' folder already exists")

    if not os.path.isdir(ds.rescaled_train_set_folder):
        os.mkdir(ds.rescaled_train_set_folder)
    else:
        print("Rescaled training sets' folder already exists")

    if not os.path.isdir(ds.rescaled_test_set_folder):
        os.mkdir(ds.rescaled_test_set_folder)
    else:
        print("Rescaled testing sets' folder already exists")

    file_list = []
    for (_, _, filenames) in os.walk(ds.training_set_folder):
        for elem in filenames:
            file_list.append(ds.training_set_folder + str(elem))

    for (_, _, filenames) in os.walk(ds.testing_set_folder):
        for elem in filenames:
            file_list.append(ds.testing_set_folder + str(elem))

    for file in file_list:
        folder = "dataset/"
        if file.startswith(folder):
            file = file[len(folder):]
        if not os.path.isfile(ds.rescaled_set_folder + file):
            rescale(folder + file, ds.rescaled_set_folder + file)
        else:
            print(os.path.basename(file) + " already rescaled")


def encode_sgrna_sequence(sequence):
    if len(sequence) > 30:
        raise Exception("Sequence is too long")
    elif len(sequence) <= 0:
        raise Exception("Sequence error (registered a sequence with negative or zero length)")
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
    dataset_file = open(dataset, 'r')
    dataser_reader = csv.reader(dataset_file, delimiter='\t')
    sequence_array = []
    efficiency_array = []
    for row in dataser_reader:
        sequence_array.append(encode_sgrna_sequence(str(row[0])))
        efficiency_array.append(float(row[1]))
    dataset_file.close()

    return sequence_array, efficiency_array


def encode_all_augmented_train_sets():
    file_list = []
    datasets_array = []
    for (_, _, filenames) in os.walk(ds.augmented_train_set_folder):
        for elem in filenames:
            file_list.append(ds.augmented_train_set_folder + str(elem))

    for file in file_list:
        sequence_array, efficiency_array = encode(file)
        datasets_array.append([sequence_array, efficiency_array, os.path.splitext(os.path.basename(file))[0]])

    return datasets_array


def encode_all_train_sets():
    file_list = []
    datasets_array = []
    for (_, _, filenames) in os.walk(ds.rescaled_train_set_folder):
        for elem in filenames:
            file_list.append(ds.rescaled_train_set_folder + str(elem))

    for file in file_list:
        sequence_array, efficiency_array = encode(file)
        datasets_array.append([sequence_array, efficiency_array, os.path.splitext(os.path.basename(file))[0]])

    return datasets_array


def encode_all_test_sets():
    file_list = []
    datasets_array = []
    for (_, _, filenames) in os.walk(ds.rescaled_test_set_folder):
        for elem in filenames:
            file_list.append(ds.rescaled_test_set_folder + str(elem))

    for file in file_list:
        sequence_array, efficiency_array = encode(file)
        datasets_array.append([sequence_array, efficiency_array])

    return datasets_array


def augment_all_train_sets():
    if not os.path.isdir(ds.augmented_set_folder):
        os.mkdir(ds.augmented_set_folder)
    else:
        print("Augmented sets' folder already exists")

    if not os.path.isdir(ds.augmented_train_set_folder):
        os.mkdir(ds.augmented_train_set_folder)
    else:
        print("Augmented training sets' folder already exists")

    file_list = []
    datasets_array = []
    for (_, _, filenames) in os.walk(ds.rescaled_train_set_folder):
        for elem in filenames:
            file_list.append(ds.rescaled_train_set_folder + str(elem))

    for file in file_list:
        dataset_file = open(file, 'r')
        dataser_reader = csv.reader(dataset_file, delimiter='\t')
        augmented_file_name = ds.augmented_train_set_folder + str(os.path.basename(file))
        if not os.path.isfile(augmented_file_name):
            augmented_dataset_file = open(augmented_file_name, 'w')
            augmented_dataset_writer = csv.writer(augmented_dataset_file, delimiter='\t', quotechar='"',
                                                  quoting=csv.QUOTE_MINIMAL)
            for row in dataser_reader:
                float(row[1])
                augmented_sequences = augment_sgrna_sequence(str(row[0]))
                for sequence in augmented_sequences:
                    augmented_dataset_writer.writerow([sequence, float(row[1])])
            augmented_dataset_file.close()
        else:
            print(os.path.basename(file) + " already augmented")

        dataset_file.close()

    return datasets_array


def augment_sgrna_sequence(sequence):
    if len(sequence) > 30:
        raise Exception("Sequence is too long")
    elif len(sequence) <= 0:
        raise Exception("Sequence error (registered a sequence with negative or zero length)")
    else:
        augmented_sequence_list = []
        nucleobasis = ["A", "C", "G", "T"]
        for base1 in nucleobasis:
            for base2 in nucleobasis:
                augmented_sequence_list.append(sequence[0:7] + base1 + base2 + sequence[9:])
        return augmented_sequence_list
