from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def baseline_model(num_pixels, num_classes):

    model = Sequential()

    model.add(
        Dense(8, input_dim=num_pixels, kernel_initializer="normal", activation="relu")
    )
    model.add(Dense(num_classes, kernel_initializer="normal"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["mean_squared_error", "accuracy"]
    )

    return model


def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):

    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype("float32")
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype("float32")

    X_train = X_train / 255
    X_test = X_test / 255


    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    model = baseline_model(num_pixels, num_classes)

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,
        batch_size=64,
        verbose=2,
    )

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f"{scores[1]:.2f}")

    return


def main():

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    trainAndPredictMLP(X_train, Y_train, X_test, Y_test)

    return


if __name__ == "__main__":
    main()
