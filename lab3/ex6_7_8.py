from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


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
    model.add(Dense(num_classes, activation="softmax"))

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
        epochs=20,
        batch_size=200,
        verbose=2,
    )

    # TODO - Application 2 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=2)
    print(f"{scores[1]:.2f}")

    return


def main():

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    trainAndPredictCNN(X_train, Y_train, X_test, Y_test)

    return


if __name__ == "__main__":
    main()
