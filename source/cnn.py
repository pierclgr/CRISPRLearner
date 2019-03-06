import tensorflow as tf
from scipy.stats import spearmanr
import numpy as np


def cnn_ontar_reg_model():
    model = tf.keras.models.Sequential()

    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(
        filters=50,
        kernel_size=(4, 4),
        strides=1,
        padding="valid",
        activation="relu"
    ))

    # Pooling Layer
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(1, 2),
        strides=1
    ))

    # Flatten Layer
    model.add(tf.keras.layers.Flatten())

    # First Fully-Connected (Dense) Layer
    model.add(tf.keras.layers.Dense(
        units=128,
        activation="relu"
    ))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(
        rate=0.3
    ))

    # Second Fully-Connected (Dense) Layer
    model.add(tf.keras.layers.Dense(
        units=128,
        activation="relu"
    ))

    # Regression Output Layer
    model.add(tf.keras.layers.Dense(
        units=1,
        activation="linear"
    ))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[r2_keras])

    return model


def prepare_input(training_set):
    features = np.array(training_set[0])
    labels = np.array(training_set[1])
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


def r2_keras(y_true, y_pred):
    ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


def spearman_metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    corr, p = spearmanr(y_true, y_pred)

