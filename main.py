from source.data_manager import *

try:
    extract_train_sets()
    extract_test_sets()
    rescale_all_sets()
    dataset_encoded = encode_train_sets()
    for elem in dataset_encoded:
        print(elem[0], elem[1])

except Exception as e:
    print(repr(e))
