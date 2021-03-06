from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from matplotlib import pyplot
import cv2
import numpy as np
from os.path import isfile
import os


def summarizeLearningCurvesPerformances(histories, scores):

    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title("Cross Entropy Loss")
        pyplot.plot(histories[i].history["loss"], color="green", label="train")
        pyplot.plot(histories[i].history["val_loss"], color="red", label="test")

        # plot accuracy
        pyplot.subplot(212)
        pyplot.title("Classification Accuracy")
        pyplot.plot(histories[i].history["accuracy"], color="green", label="train")
        pyplot.plot(histories[i].history["val_accuracy"], color="red", label="test")

        # print accuracy for each split
        print("Accuracy for set {} = {}".format(i, cores[i]))

    pyplot.show()

    print(
        "Accuracy: mean = {:.3f} std = {:.3f}, n = {}".format(
            mean(scores) * 100, std(scores) * 100, len(scores)
        )
    )


def prepareData(trainX, trainY, testX, testY):

    # TODO - Application 1 - Step 3 - reshape the data to be of size [samples][width][height][channels]

    trainX = np.reshape(trainX, (*trainX.shape, 1))
    testX = np.reshape(testX, (*testX.shape, 1))

    # TODO - Application 1 - Step 4 - normalize the input values
    trainX = 1 / 255 * trainX.astype("float32")
    testX = 1 / 255 * testX.astype("float32")

    # TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix
    trainY = np_utils.to_categorical(trainY)
    testY = np_utils.to_categorical(testY)

    return trainX, trainY, testX, testY


def defineModel(input_shape, num_classes):

    # Application 1 - Step 6 - Initialize the sequential model
    model = Sequential()

    # TODO - Application 1 - Step 6 - Create the first hidden layer as a convolutional layer
    model.add(
        Conv2D(
            32,
            3,
            input_shape=input_shape,
            activation="relu",
            kernel_initializer="he_uniform",
        )
    )

    # TODO - Application 1 - Step 6 - Define the pooling layer
    model.add(MaxPooling2D())

    # TODO - Application 1 - Exercise 6 - Add a dropout layer
    model.add(Dropout(0.1))

    # TODO - Application 1 - Step 6 - Define the flatten layer
    model.add(Flatten())

    # TODO - Application 1 - Step 6 - Define a dense layer of size 16
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))

    # TODO - Application 1 - Step 6 - Define the output layer
    model.add(Dense(num_classes, activation="softmax"))

    # TODO - Application 1 - Step 6 - Compile the model
    model.compile(
        optimizer=SGD(0.01, 0.9), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def trainAndEvaluateClassic(trainX, trainY, testX, testY):

    accuracy = 0

    # Application 1 - Call the defineModel function
    weights_name = './Fashion_MNIST_model.h5'

    if not isfile(weights_name):
      acc = trainAndEvaluateClassic(trainX, trainY, testX, testY)
      model.save_weights(weights_name)
    else:
      model.load_weights(weights_name)

    model = defineModel((28, 28, 1), 10)

    # TODO - Application 1 - Step 7 - Train the model
    print(trainX.shape)
    model.fit(trainX, trainY, batch_size=32, epochs=10)

    # TODO - Application 1 - Step 7 - Evaluate the model
    accuracy = model.evaluate(testX, testY)

    return accuracy


def trainAndEvaluateKFolds(trainX, trainY, testX, testY):

    k_folds = 5

    scores = []
    histories = []

    # Application 2 - Prepare the cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    # Enumerate splits
    for train_idx, val_idx in kfold.split(trainX):

        # TODO - Application 2 - Step 1 - Select data for train and validation

        # TODO - Application 2 - Step 1 - Create the model

        # TODO - Application 2 - Step 1 - Fit the model

        # TODO - Application 2 - Step 1 - Evaluate the model on the test dataset

        # TODO - Application 2 - Step 1 - Save the accuracy scores in the scores list
        # and the learning history in the histories list

        # Delete this
        continue

    return scores, histories


def main():

    # TODO - Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    # TODO - Application 1 - Step 2 - Print the size of the train/test dataset
    print("Train: X={}, Y={}".format(trainX.shape, trainY.shape))
    print("Test: X={}, Y={}".format(testX.shape, testY.shape))

    # TODO - Application 1 - Call the prepareData method
    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)



    # TODO - Application 2 Train and evaluate the model using K-Folds strategy

    # Application 2 - Step2 - System performance presentation
    # summarizeLearningCurvesPerformances(histories, scores)

    return


if __name__ == "__main__":
    main()
