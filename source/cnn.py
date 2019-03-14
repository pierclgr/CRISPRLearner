import tensorflow as tf
import numpy as np
import dataset.paths as ds
import os
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from source.data_manager import extract_train_sets, extract_test_sets, rescale_all_sets, augment_all_train_sets, \
    encode_all_augmented_train_sets


def cnn_ontar_reg_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(
        input_shape=(4, 30, 1)
    ))

    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(
        filters=50,
        kernel_size=(4, 4),
        strides=1,
        padding="same"
    ))

    # ReLU Activation Layer
    model.add(tf.keras.layers.Activation(activation="relu"))

    # Pooling Layer
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(1, 2),
        strides=1
    ))

    # Flatten Layer
    model.add(tf.keras.layers.Flatten())

    # First Fully-Connected (Dense) Layer
    model.add(tf.keras.layers.Dense(
        units=128
    ))

    # ReLU Activation Layer
    model.add(tf.keras.layers.Activation(activation="relu"))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(
        rate=0.3
    ))

    # Second Fully-Connected (Dense) Layer
    model.add(tf.keras.layers.Dense(
        units=128
    ))

    # ReLU Activation Layer
    model.add(tf.keras.layers.Activation(activation="relu"))

    # Output Layer
    model.add(tf.keras.layers.Dense(
        units=1
    ))

    # Linear Regression Layer
    model.add(tf.keras.layers.Activation(activation="linear"))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")

    return model


def prepare_set(set):
    features = np.array(set[0])
    labels = np.array(set[1])
    features = np.reshape(features, features.shape + (1,))
    labels = np.array(labels)
    return features, labels, set[2]


def train_model(dataset, save_weigths=True):
    features, labels, name = prepare_set(dataset)
    features, labels = shuffle(features, labels)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, mode="min", verbose=0)

    model = cnn_ontar_reg_model()
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.1,
                                                                              shuffle=True)
    results = model.fit(x=train_features, y=train_labels, batch_size=40, epochs=250, verbose=1,
                        validation_data=(val_features, val_labels), callbacks=[early_stopping])
    loss = model.evaluate(x=val_features, y=val_labels, verbose=1)
    y_pred = model.predict(x=val_features)
    spearman, _ = spearmanr(val_labels, y_pred)
    print("Spearman of", name, ":", spearman)
    print("Loss of", name, ":", loss)
    if save_weigths is True:
        save_model_weights(model, name)
    return model


def train_on_all_train_sets():
    extract_train_sets()
    extract_test_sets()
    rescale_all_sets()
    augment_all_train_sets()
    train_sets_array = encode_all_augmented_train_sets()
    for dataset in train_sets_array:
        train_model(dataset)


def save_model_weights(model, name):
    if not os.path.isdir(ds.model_weights_folder):
        os.mkdir(ds.model_weights_folder)
    model.save_weights(ds.model_weights_folder + name + "_weights.hdf5")


def load_model_weights(model, name):
    model.load_weights(ds.model_weights_folder + name + "_weights.hdf5")
    return model
