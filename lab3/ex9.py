from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def CNN_model(input_shape, num_classes):

    model = Sequential()

    model.add(Conv2D(30, 5, input_shape=(28, 28, 1), activation="relu"))

    model.add(MaxPooling2D())

    model.add(Conv2D(15, 3, activation="relu"))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))

    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

    X_train = X_train / 255
    X_test = X_test / 255

    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    model = CNN_model((28, 28, 1), num_classes)

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=200,
        verbose=2,
    )

    scores = model.evaluate(X_test, Y_test, verbose=2)
    print(f"{scores[1]:.2f}")

    return


def main():

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    trainAndPredictCNN(X_train, Y_train, X_test, Y_test)

    return


if __name__ == "__main__":
    main()
