import numpy as np
import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


def define_and_model():

    model = Sequential()

    weights = np.array([
        2,
        2
    ]).reshape(2, 1)

    bias = np.array([
        -3
    ])

    model.add(
        Dense(
            1,
            input_dim=2,
            weights=[weights, bias],
            activation="sigmoid",
        )
    )

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

    return model


def define_xor_model():

    model = Sequential()

    weights_l1 = np.array([[2, -1], [2, -1]])

    bias_l1 = np.array([-1, 1.5])

    weights_l2 = np.array([1, 1]).reshape(2, 1)

    bias_l2 = np.array([-1.5])

    model.add(Dense(2, input_dim=2, activation="tanh", weights=[weights_l1, bias_l1]))
    model.add(
        Dense(1, input_dim=2, activation="sigmoid", weights=[weights_l2, bias_l2])
    )

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

    return model


def train_and():

    model = define_and_model()

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "uint8")
    y = np.array([0, 0, 0, 1], "uint8")

    model.fit(x, y, epochs=3000, verbose=0)

    predictions = model.predict(x)

    print(predictions)


def train_xor():

    model = define_xor_model()

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "uint8")
    y = np.array([0, 1, 1, 0], "uint8")

    model.fit(x, y, epochs=3000, verbose=0)

    predictions = model.predict(x)

    print(predictions)


if __name__ == "__main__":
    train_xor()