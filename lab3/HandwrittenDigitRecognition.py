from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def baseline_model(num_pixels, num_classes):

    # Application 1 - Step 5 - Initialize the sequential model
    model = Sequential()

    # TODO - Application 1 - Step 5 - build a standard feed-forward network with one dense hidden layer(with 8 neurons) and one dense output layer
    model.add(
        Dense(8, input_dim=num_pixels, kernel_initializer="normal", activation="relu")
    )
    model.add(Dense(num_classes, kernel_initializer="normal"))

    # TODO - Application 1 - Step 6 - Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):

    # TODO - Application 1 - Step 2 - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)

    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype("float32")
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype("float32")

    # TODO - Application 1 - Step 3 - Normalize the input values

    X_train = X_train / 255
    X_test = X_test / 255

    # TODO - Application 1 - Step 4 - Transform the classes labels into a binary matrix

    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    # Application 1 - Step 5 - Call the baseline_model function
    model = baseline_model(num_pixels, num_classes)

    # TODO - Application 1 - Step 7 - Train the model
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=200,
        verbose=2,
    )

    # TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    # print(f"Baseline Error: {(100 * (1 - scores)):.2f}")
    print(scores)

    return


def CNN_model(input_shape, num_classes):

    # Application 2 - Step 6 - Initialize the sequential model
    model = Sequential()

    # TODO - Application 2 - Step 6 - Create the first hidden layer as a convolutional layer
    model.add(Conv2D(8, 3, activation="relu"))

    # TODO - Application 2 - Step 6 - Define the pooling layer
    model.add(MaxPooling2D())

    # TODO - Application 2 - Step 6 - Define the Dropout layer
    model.add(Dropout(0.2))

    # TODO - Application 2 - Step 6 - Define the flatten layer
    model.add(Flatten())

    # TODO - Application 2 - Step 6 - Define a dense layer of size 128
    model.add(Dense(128, activation="relu"))

    # TODO - Application 2 - Step 6 - Define the output layer
    model.add(Dense(10, activation="softmax"))

    # TODO - Application 2 - Step 7 - Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    # TODO - Application 2 - Step 3 - reshape the data to be of size [samples][width][height][channels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

    # TODO - Application 2 - Step 4 - normalize the input values

    X_train = X_train / 255
    X_test = X_test / 255

    # TODO - Application 2 - Step 5 - Transform the classes labels into a binary matrix

    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    # Application 2 - Step 6 - Call the cnn_model function
    model = CNN_model((28, 28, 1), num_classes)

    # TODO - Application 2 - Step 8 - Train the model
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=200,
        verbose=2,
    )

    # TODO - Application 2 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=2)
    print(scores)
    # print(f"Baseline Error: {(100 * (1 - scores)):.2f}")

    return


def main():

    # TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # TODO - Application 1 - Step 2 - Train and predict on a MLP
    # trainAndPredictMLP(X_train, Y_train, X_test, Y_test)

    # TODO - Application 2 - Train and predict on a CNN
    trainAndPredictCNN(X_train, Y_train, X_test, Y_test)

    return


if __name__ == "__main__":
    main()
