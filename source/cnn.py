# Import libraries and modules
import tensorflow as tf
import numpy as np
import dataset.paths as ds
import os
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from source.data_manager import extract_all_sets, rescale_all_sets, augment_all_sets, \
    encode_all_augmented_sets, encode_all_sets
import matplotlib.pyplot as plt


def cnn_ontar_reg_model():
    """
    Creates a regression model for on-target efficiency prediction

    :return: the created regression model
    """

    # Sequential model
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.InputLayer(
        input_shape=(4, 23, 1)
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

    # Compile the model using mean squared error as loss function and Adam as optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")

    return model


def prepare_set(dataset):
    """
    Prepares the given dataset for training task performing reshapes

    :param dataset: dataset to reshape
    :return: a list of features, a list of labels and dataset name
    """

    features = np.array(dataset[0])
    labels = np.array(dataset[1])
    features = np.reshape(features, features.shape + (1,))

    return features, labels, dataset[2]


def train_model(dataset, save_weigths=True, verbose=1):
    """
    Trains a cnn regression model using the given dataset

    :param dataset: training set
    :param save_weigths: boolean to indicate if the user wants to save model weights or not
    :param verbose: verbose mode
    :return: trained model
    """

    # Prepare the given training set for training
    features, labels, name = prepare_set(dataset)

    # Randomly shuffle the training set
    features, labels = shuffle(features, labels)

    print("Training on", name)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, mode="min", verbose=verbose)

    # Generate a cnn regression model
    model = cnn_ontar_reg_model()

    # Split the training set into training and validation set (80 % / 20 %)
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2,
                                                                              shuffle=True)

    # Start training
    history = model.fit(x=train_features, y=train_labels, epochs=250, verbose=verbose,
                        validation_data=(val_features, val_labels), callbacks=[early_stopping])

    # Evaluate model using validation set
    loss = model.evaluate(x=val_features, y=val_labels, verbose=verbose)

    # Plot training history
    plot_history(history, name)

    # Predict efficiencies for validation set
    y_pred = model.predict(x=val_features)

    # Calculate spearman correlation score between predicted and real efficiencies
    spearman, _ = spearmanr(val_labels, y_pred)

    # Print score and loss of the model
    print("Spearman of", name, ":", spearman)
    print("Loss of", name, ":", loss, "\n")

    # If the user specified it, save model weights
    if save_weigths:
        save_model_weights(model, name)

    return model


def train_all_models(verbose=1, save=True):
    """
    Trains models using all training sets in the training sets folder

    :param verbose: verbose mode
    :param save: specifies if user wants to save model weights or not
    :return: None
    """

    # Extract all sets from Haeussler, saving them in training sets folder
    extract_all_sets()

    # Rescale all sets in training sets folder
    rescale_all_sets()

    """
    # Augment all sets in rescaled training sets folder
    augment_all_sets()

    # Encode all sets in augmented training sets folder
    train_sets_array = encode_all_augmented_sets()
    """

    train_sets_array = encode_all_sets()

    # For each augmented training set
    for dataset in train_sets_array:
        # Train a model using the current dataset
        train_model(dataset=dataset, save_weigths=save, verbose=verbose)


def save_model_weights(model, name):
    """
    Saves the specified model weights

    :param model: model which weights are going to be saved
    :param name: name of the model
    :return: None
    """

    # Create weights folder if it does not exist
    if not os.path.isdir(ds.model_weights_folder):
        os.mkdir(ds.model_weights_folder)

    # Save given model weights
    model.save_weights(ds.model_weights_folder + name + ".hdf5")


def load_model_weights(model, name):
    """
    Loads weights to the specified model

    :param model: model which weights are going to be loaded
    :param name: name of the model used for loading weights
    :return: model with weights loaded
    """

    # Load given model weight
    model.load_weights(ds.model_weights_folder + name + ".hdf5")

    return model


def plot_history(history, dataset):
    """
    Plots and saves model history

    :param history: history of model to plot
    :param dataset: dataset name
    :return: None
    """

    # Create histories folder if it does not exist
    if not os.path.isdir(ds.model_histories_folder):
        os.mkdir(ds.model_histories_folder)

    # Plot and save history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(dataset + " loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(ds.model_histories_folder + dataset + ".png")
    plt.clf()
