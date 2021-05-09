from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.applications import VGG16

import os
import shutil


def prepareDatabase(original_directory, base_directory):

    # If the folder already exist remove everything
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)

    # Recreate the basefolder
    os.mkdir(base_directory)

    # TODO - Application 1 - Step 1 - Struncture the dataset in training, validation and testing directories

    train_directory = os.path.join(base_directory, "train")
    os.mkdir(train_directory)
    validation_directory = os.path.join(base_directory, "validation")
    os.mkdir(validation_directory)
    test_directory = os.path.join(base_directory, "test")
    os.mkdir(test_directory)

    # TODO - Application 1 - Step 1 - Create the cat/dog training directories - See figure 4

    # create the cat/dog training directories
    train_cats_directory = os.path.join(train_directory, "cats")
    os.mkdir(train_cats_directory)
    train_dogs_directory = os.path.join(train_directory, "dogs")
    os.mkdir(train_dogs_directory)

    # TODO - Application 1 - Step 1 - Create the cat/dog validation directories - See figure 4

    # create the cat/dog validation directories
    validation_cats_directory = os.path.join(validation_directory, "cats")
    os.mkdir(validation_cats_directory)
    validation_dogs_directory = os.path.join(validation_directory, "dogs")
    os.mkdir(validation_dogs_directory)

    # TODO - Application 1 - Step 1 - Create the cat/dog testing directories - See figure 4

    # create the cat/dog testing directories
    test_cats_directory = os.path.join(test_directory, "cats")
    os.mkdir(test_cats_directory)
    test_dogs_directory = os.path.join(test_directory, "dogs")
    os.mkdir(test_dogs_directory)

    # TODO - Application 1 - Step 1 - Copy the first 1000 cat images in to the training directory (train_cats_directory)

    # copy the first 1000 cat images in to the training directory (train_cats_directory)
    original_directory_cat = os.path.join(original_directory, 'cats')
    fnames = ['{}.jpg'.format(i) for i in range(1000)]

    for fname in fnames:
        src = os.path.join(original_directory_cat, fname)
        dst = os.path.join(train_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 1 - Copy the next 500 cat images in to the validation directory (validation_cats_directory)

    fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
    
    for fname in fnames:
        src = os.path.join(original_directory_cat, fname)
        dst = os.path.join(validation_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 1 - Copy the next 500 cat images in to the test directory (test_cats_directory)

    fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]
    
    for fname in fnames:
        src = os.path.join(original_directory_cat, fname)
        dst = os.path.join(test_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 2 - Copy the first 1000 dogs images in to the training directory (train_dogs_directory)

    original_directory_dog = os.path.join(original_directory, 'dogs')
    fnames = ['{}.jpg'.format(i) for i in range(1000)]

    for fname in fnames:
        src = os.path.join(original_directory_dog, fname)
        dst = os.path.join(train_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 2 - Copy the next 500 dogs images in to the validation directory (validation_dogs_directory)

        fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]

    for fname in fnames:
        src = os.path.join(original_directory_dog, fname)
        dst = os.path.join(validation_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 2 - Copy the next 500 dogs images in to the test directory (test_dogs_directory)

    fnames = ['{}.jpg'.format(i) for i in range(1500, 2000)]

    for fname in fnames:
        src = os.path.join(original_directory_dog, fname)
        dst = os.path.join(test_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Step 1 - As a sanitary check verify how many pictures are in each directory

    # As a sanitary check verify how many pictures are in each directory
    print('Total number of CATS used for training = {}'.format(len(os.listdir(train_cats_directory))))
    print('Total number of CATS used for validation = {}'.format(len(os.listdir(validation_cats_directory))))
    print('Total number of CATS used for testing = {}'.format(len(os.listdir(test_cats_directory))))
    print('Total number of DOGS used for training = {}'.format(len(os.listdir(train_dogs_directory))))
    print('Total number of DOGS used for validation = {}'.format(len(os.listdir(validation_dogs_directory))))
    print('Total number of DOGS used for testing = {}'.format(len(os.listdir(test_dogs_directory))))

    return


def defineCNNModelFromScratch():

    # Application 1 - Step 3 - Initialize the sequential model
    model = models.Sequential()

    # TODO - Application 1 - Step 3 - Create the first hidden layer as a convolutional layer

    # TODO - Application 1 - Step 3 - Define a pooling layer

    # TODO - Application 1 - Step 3 - Create the third hidden layer as a convolutional layer

    # TODO - Application 1 - Step 3 - Define a pooling layer

    # TODO - Application 1 - Step 3 - Create another convolutional layer

    # TODO - Application 1 - Step 3 - Define a pooling layer

    # TODO - Application 1 - Step 3 - Create another convolutional layer

    # TODO - Application 1 - Step 3 - Define a pooling layer

    # TODO - Application 1 - Step 3 - Define the flatten layer

    # TODO - Application 1 - Step 3 - Define a dense layer of size 512

    # TODO - Application 1 - Step 3 - Define the output layer

    # TODO - Application 1 - Step 3 - Visualize the network arhitecture (list of layers)

    # TODO - Application 1 - Step 3 - Compile the model

    return model


def defineCNNModelVGGPretrained():

    # TODO - Exercise 6 - Load the pretrained VGG16 network in a variable called baseModel
    # The top layers will be omitted; The input_shape will be kept to (150, 150, 3)

    # TODO - Exercise 6 - Visualize the network arhitecture (list of layers)

    # TODO - Exercise 6 - Freeze the baseModel layers to not to allow training

    # Create the final model and add the layers from the baseModel
    VGG_model = models.Sequential()
    # VGG_model.add(baseModel)

    # TODO - Exercise 6 - Add the flatten layer

    # TODO - Exercise 6 - Add the dropout layer

    # TODO - Exercise 6 - Add a dense layer of size 512

    # TODO - Exercise 6 - Add the output layer

    # TODO - Exercise 6 - Compile the model

    return VGG_model


def imagePreprocessing(base_directory):

    train_directory = base_directory + "/train"
    validation_directory = base_directory + "/validation"

    # TODO - Application 1 - Step 2 - Create the image data generators for train and validation

    # TODO - Application 1 - Step 2 - Analize the output of the train and validation generators

    # return train_generator, validation_generator


def visualizeTheTrainingPerformances(history):

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    pyplot.title("Training and validation accuracy")
    pyplot.plot(epochs, acc, "bo", label="Training accuracy")
    pyplot.plot(epochs, val_acc, "b", label="Validation accuracy")
    pyplot.legend()

    pyplot.figure()
    pyplot.title("Training and validation loss")
    pyplot.plot(epochs, loss, "bo", label="Training loss")
    pyplot.plot(epochs, val_loss, "b", label="Validation loss")
    pyplot.legend

    pyplot.show()

    return


def main():

    original_directory = "../Kaggle_Cats_And_Dogs_Dataset"
    base_directory = "Kaggle_Cats_And_Dogs_Dataset_Small"

    prepareDatabase(original_directory, base_directory)

    # TODO - Application 1 - Step 1 - Load the Fashion MNIST dataset in Keras

    # TODO - Application 1 - Step 2 - Call the imagePreprocessing method

    # TODO - Application 1 - Step 3 - Call the method that creates the CNN model

    # TODO - Application 1 - Step 4 - Train the model

    # TODO - Application 1 - Step 5 - Visualize the system performance using the diagnostic curves

    return


if __name__ == "__main__":
    main()
