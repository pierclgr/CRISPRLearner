# Import libraries and modules
from source.cnn import train_all_models, train_model, cnn_ontar_reg_model, load_model_weights
from source.data_manager import augment, encode, rescale, encode_sgrna_sequence
import dataset.paths as ds
import os
import re
from shutil import copyfile
import numpy as np
import string


def print_menu():
    """
    Prints the main cli menu

    :return: None
    """

    # Print main cli menu
    print("\nWhat do you want to do?")
    print("\n\t1) Train models using all training sets")
    print("\t2) Train a model using your dataset")

    # If some model's weights have been saved, print the option for predicting a sequence efficiency
    if os.path.isdir(ds.model_weights_folder):
        print("\t3) Predict a sequence efficiency")

    print("\n\t9) Help")
    print("\t0) Quit\n")
    selection = input()
    dispatch_selection(selection)
    while selection is not "0":
        print("\nWhat do you want to do?")
        print("\n\t1) Train models using all training sets")
        print("\t2) Train a model using your dataset")
        if os.path.isdir(ds.model_weights_folder):
            print("\t3) Predict a sequence efficiency")

        print("\n\t9) Help")
        print("\t0) Quit\n")
        selection = input()
        dispatch_selection(selection)


def dispatch_selection(selection):
    """
    Performs command given from the user

    :param selection: command to perform
    :return: None
    """

    # If user has selected option "Train all models using all training sets"
    if selection == "1":

        # Ask if user wants to save model weights
        print("Do you want to save model weights? (Y/N)")
        answer = str(input())
        answer = answer.upper()
        2
        while answer != "Y" and answer != "N":
            print("Answer not valid, insert a valid answer (Y/N)")
            answer = str(input())
            answer = answer.upper()

        # Save weights if users wants to
        if answer == "Y":
            train_all_models(save=True)
        elif answer == "N":
            train_all_models(save=False)

    # If user has selected option "Train your model using your dataset"
    elif selection == "2":

        # Ask user to insert the dataset path
        print("Insert your dataset path")
        path = input()

        # Create training set folder if it does not exist
        if not os.path.isdir(ds.training_set_folder):
            os.mkdir(ds.training_set_folder)

        # Copy user's dataset into training sets' folder if it does not exist
        if not os.path.isfile(ds.training_set_folder + os.path.basename(path)):
            copyfile(path, ds.training_set_folder + os.path.basename(path))

        # Create rescaled sets' folder if it does not exist
        if not os.path.isdir(ds.rescaled_set_folder):
            os.mkdir(ds.rescaled_set_folder)

        # Create rescaled training sets' folder if it does not exist
        if not os.path.isdir(ds.rescaled_train_set_folder):
            os.mkdir(ds.rescaled_train_set_folder)

        # Rescale dataset given by user
        rescaled_set_file = ds.rescaled_train_set_folder + os.path.basename(path)
        rescale(ds.training_set_folder + os.path.basename(path))

        # Create augmented sets' folder if it does not exist
        if not os.path.isdir(ds.augmented_set_folder):
            os.mkdir(ds.augmented_set_folder)

        # Create augmented training sets' folder if it does not exist
        if not os.path.isdir(ds.augmented_train_set_folder):
            os.mkdir(ds.augmented_train_set_folder)

        # Augment dataset given by user
        augmented_set_file = ds.augmented_train_set_folder + os.path.basename(path)
        augment(rescaled_set_file)

        # Encode dataset given by user
        sequence_array, efficiency_array = encode(augmented_set_file)
        encoded_set = [sequence_array, efficiency_array, os.path.splitext(os.path.basename(augmented_set_file))[0]]

        # Ask if user wants to save model weights
        print("Do you want to save model weights? (Y/N)")
        answer = str(input())
        answer = answer.upper()
        while answer != "Y" and answer != "N":
            print("Answer not valid, insert a valid answer (Y/N)")
            answer = str(input())
            answer = answer.upper()

        # Save weights if users wants to
        if answer == "Y":
            train_model(dataset=encoded_set, save_weigths=True)
        elif answer == "N":
            train_model(dataset=encoded_set, save_weigths=False)

    # If user has selected option "Predict sequence efficiency"
    elif selection == "3":

        # If some model weights have been saved
        if os.path.isdir(ds.model_weights_folder):

            print("Select the model associated to cell and species that you want to use:\n")

            # For each model in weights folder
            for (_, _, filenames) in os.walk(ds.model_weights_folder):

                i = 0

                # For each model in weights folder
                while i < len(filenames):

                    file_name = os.path.splitext(filenames[i])[0]

                    # Add model specifications based on model name
                    if file_name == os.path.splitext(ds.varshney)[0]:
                        model_specs = "(Zebrafish)"
                    elif file_name == os.path.splitext(ds.gandhi_ci2)[0]:
                        model_specs = "(Ciona)"
                    elif file_name == os.path.splitext(ds.moreno_mateos)[0]:
                        model_specs = "(Zebrafish)"
                    elif file_name == os.path.splitext(ds.gagnon)[0]:
                        model_specs = "(Zebrafish)"
                    elif file_name == os.path.splitext(ds.chari_293t)[0]:
                        model_specs = "(Human, HEK293T cell line)"
                    elif file_name == os.path.splitext(ds.hart_hct116)[0]:
                        model_specs = "(Human, HCT116 cell line)"
                    elif file_name == os.path.splitext(ds.doench_mel4)[0]:
                        model_specs = "(Mouse, EL4 cell line)"
                    elif file_name == os.path.splitext(ds.wangxu_hl60)[0]:
                        model_specs = "(Human, HL60 cell line)"
                    elif file_name == os.path.splitext(ds.farboud)[0]:
                        model_specs = "(C. elegans)"
                    elif file_name == os.path.splitext(ds.doench_hg19)[0]:
                        model_specs = "(Human, A375 cell line)"
                    else:
                        model_specs = ""

                    # Print selectable models
                    print("\t" + str(i + 1) + ")", file_name, model_specs)

                    i += 1

                print()

                # Let user choose model that he wants to use
                model_selection = int(input())
                while model_selection < 1 or model_selection > i:
                    print("Selected model is not valid, insert a correct model number")
                    model_selection = int(input())

                # Ask user for a sgRNA sequence
                print("Insert your sgRNA sequence (max 30 length)")
                sequence = str(input())
                sequence = sequence.upper()

                # If given sequence isn't valid or it's too long, print an error and ask for a new sequence
                while not re.match("^[ACGT]+$", str(sequence)) or len(sequence) > 30:
                    print("The sgRNA sequence you wrote is not valid or is too long, insert a valid "
                          "sgRNA sequence (max 30 length)")
                    sequence = str(input())
                    sequence = sequence.upper()

                # Create cnn regression model
                model = cnn_ontar_reg_model()

                # Load selected model weights
                load_model_weights(model, os.path.splitext(filenames[model_selection - 1])[0])

                # Encode sequence and reshape for prediction
                sequence = encode_sgrna_sequence(sequence)
                sequence = np.reshape(sequence, (1,) + sequence.shape + (1,))

                # Predict given sequence efficiency
                prediction = model.predict(x=sequence)

                print("Predicted efficiency is", prediction[0][0])

        # Else print an error
        else:
            print("\nAction selected not valid, insert a valid action\n")

    # If user has selected option "Help"
    elif selection == "9":

        # Print command help
        print("\n"
              "Selecting command \"1) Train models using all training sets\", the system extracts 10 default\n"
              "training sets from Haeussler dataset file (if they aren't already extracted) and trains a single\n"
              "model for each set in the training sets folder, including dataset added by user using command \"2)\".\n"
              "Before performing the train task, the system rescales, augments and encodes all datasets in training\n"
              "sets folder.")

        print("\n"
              "Selecting command \"2) Train a model using your dataset\", the system allows the user to load a\n"
              "custom dataset and train a model using this dataset. Before the training task, our system copies\n"
              "user dataset into training sets folder, making it available for future calls of \"2)\" command.\n"
              "Before performing the train task, the system rescales, augments and encodes the given dataset.")

        # If some model weights have been saved
        if os.path.isdir(ds.model_weights_folder):

            print("\n"
                  "Selecting command \"3) Predict a sequence efficiency\", the system allows the user to insert a\n"
                  "sequence and predict it's efficiency using the current saved model weights. If no model weights\n"
                  "are currently saved in the weights folder, this command will not be available until the user\n "
                  "performs a training task.")

        print("\n"
              "Selecting command \"9) Help\", the system gives an explanation of all commands.")

        print("\n"
              "Selecting command \"0) Quit\", the system is closed.")

    # If user has selected option "Quit"
    elif selection == "0":

        # Do nothing
        pass

    # Else, print an error
    else:
        print("\nAction selected not valid, insert a valid action\n")
