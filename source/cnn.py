import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr


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
        padding="valid"
    ))

    # ReLU Activation Layer
    model.add(tf.keras.layers.Activation("relu"))

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
    model.add(tf.keras.layers.Activation("relu"))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(
        rate=0.3
    ))

    # Second Fully-Connected (Dense) Layer
    model.add(tf.keras.layers.Dense(
        units=128
    ))

    # ReLU Activation Layer
    model.add(tf.keras.layers.Activation("relu"))

    # Output Layer
    model.add(tf.keras.layers.Dense(
        units=1
    ))

    # Linear Regression Layer
    model.add(tf.keras.layers.Activation("linear"))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")

    return model


def prepare_set(set):
    features = np.array(set[0])
    labels = np.array(set[1])
    features = np.reshape(features, features.shape + (1,))
    labels = np.array(labels)
    return features, labels


def pearson(y_true, y_pred):
    # normalise
    n_y_true = (y_true - tf.keras.backend.mean(y_true[:])) / tf.keras.backend.std(y_true[:])
    n_y_pred = (y_pred - tf.keras.backend.mean(y_pred[:])) / tf.keras.backend.std(y_pred[:])

    top = tf.keras.backend.sum(
        (n_y_true[:] - tf.keras.backend.mean(n_y_true[:])) * (n_y_pred[:] - tf.keras.backend.mean(n_y_pred[:])),
        axis=[-1, -2])
    bottom = tf.keras.backend.sqrt(
        tf.keras.backend.sum(tf.keras.backend.pow((n_y_true[:] - tf.keras.backend.mean(n_y_true[:])), 2),
                             axis=[-1, -2]) * tf.keras.backend.sum(
            tf.keras.backend.pow(n_y_pred[:] - tf.keras.backend.mean(n_y_pred[:]), 2), axis=[-1, -2]))

    result = top / bottom

    return tf.keras.backend.mean(result)


def r2(y_true, y_pred):
    ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
