from source.data_manager import extract_train_sets, extract_test_sets, rescale_all_sets, encode_all_sets
from source.cnn import cnn_ontar_reg_model, prepare_input
import sys
import os
from sklearn.model_selection import KFold

try:
    extract_train_sets()
    extract_test_sets()
    rescale_all_sets()

    datasets_array = encode_all_sets()

    model = cnn_ontar_reg_model(learning_rate=0.001)
    epochs = 250
    batch_size = 32

    inner_cv = KFold(n_splits=10)
    outer_cv = KFold(n_splits=10)

    for elem in datasets_array:
        if len(elem[0]) == len(elem[1]):
            features, labels = prepare_input(elem)
            outer_cv = KFold(n_splits=10)
            inner_cv = KFold(n_splits=10)
            for trainval_index, test_index in outer_cv.split(features, labels):
                trainval_features, test_features, trainval_labels, test_labels = features[trainval_index], features[
                    test_index], labels[trainval_index], labels[test_index]
                for train_index, val_index in inner_cv.split(trainval_features, trainval_labels):
                    train_features, val_features, train_labels, val_labels = trainval_features[train_index], \
                                                                             trainval_features[val_index], \
                                                                             trainval_labels[train_index], \
                                                                             trainval_labels[val_index]
                    model.fit(x=train_features, y=train_labels, epochs=epochs, verbose=0, batch_size=batch_size,
                              validation_data=(val_features, val_labels))

                model.evaluate(x=test_features, y=test_labels, verbose=1)
        # results = model.fit(x=features, y=labels, epochs=epochs, verbose=1, batch_size=batch_size)

    else:
        raise Exception("Sequence set and efficiency set are not the same size")
except Exception as e:
    print(repr(e))
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
