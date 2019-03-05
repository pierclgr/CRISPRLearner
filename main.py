from source.data_manager import *
from source.cnn import *

try:
    extract_train_sets()
    extract_test_sets()
    rescale_all_sets()

    datasets_array = encode_all_sets()

    model = cnn_ontar_reg_model()

    for elem in datasets_array:
        if len(elem[0]) == len(elem[1]):
            features, labels = prepare_input(elem)
            
        else:
            raise Exception("Sequence set and efficiency set are not the same size")
except Exception as e:
    print(repr(e))
