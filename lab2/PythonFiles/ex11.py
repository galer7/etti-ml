import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.metrics import accuracy_score


# TODO - Application 3 - Step 5 - Create the ANN model
def modelDefinition():

    # TODO - Application 3 - Step 5a - Define the model as a Sequential model
    model = Sequential()

    # TODO - Application 3 - Step 5b - Add a Dense layer with 13 neurons to the model
    model.add(Dense(8, input_dim=13, kernel_initializer="normal", activation="relu"))
    model.add(Dense(16, input_dim=8, kernel_initializer="normal", activation="relu"))

    # TODO - Application 3 - Step 5c - Add a Dense layer (output layer) with 1 neuron
    model.add(Dense(1, kernel_initializer="normal"))

    # TODO - Application 3 - Step 5d - Compile the model by choosing the optimizer(adam) ant the loss function (MSE)
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


def main():

    # TODO - Application 3 - Step 1 - Read data from "Houses.csv" file
    data = pd.read_csv("./Houses.csv").values

    # TODO - Application 3 - Step 2 - Shuffle the data
    np.random.shuffle(data)

    # TODO - Application 3 - Step 3 - Separate the data from the labels (x_data / y_data)
    x_data = data[:, [i for i in range(14) if i != 4]]
    y_data = data[:, 4]

    print(x_data.shape)
    print(y_data.shape)

    # TODO - Application 3 - Step 4 - Separate the data into training/testing dataset
    eigthy_percent = int(0.8 * len(x_data))

    x_train = x_data[:eigthy_percent, :]
    y_train = y_data[:eigthy_percent]

    print(x_train.shape)
    print(y_train.shape)

    x_test = x_data[eigthy_percent:, :]
    y_test = y_data[eigthy_percent:]

    # TODO - Application 3 - Step 5 - Call the function "modelDefinition"
    model = modelDefinition()

    # TODO - Application 3 - Step 6 - Train the model for 100 epochs and a batch of 16 samples
    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2)

    # Step 7
    predictions = model.predict(x_test)

    # TODO - Exercise 8 - Compute the MSE for the test data
    mse = 1 / len(x_test) * np.sum(np.square(predictions - y_test))
    print(f"Mean Square Error = {mse}")  # Mean Square Error = 18099.76277140959

    return


if __name__ == "__main__":
    main()