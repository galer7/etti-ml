import pandas as pd
import numpy as np
import random

def get_iris_data(path):
    df = pd.read_csv(path)
    headers = [
        'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Class'
    ]

    x = []
    y = []

    for row in [x[1] for x in df.iterrows()]:
        xs = []

        for id_h, h in enumerate(headers):
            if id_h == len(headers) - 1:
                y.append(row[h])
                x.append(xs)
            else:
                xs.append(row[h])

    d = {}
    cnt = 0

    for label in y:
        if label not in d:
            d[label] = cnt
            cnt += 1

    for id, label in enumerate(y):
        y[id] = d[label]

    print(y)

    return x, y


def predict(xsample, W):

    
    s = []
    s = W.dot(xsample)

    return s


# TODO - Application 3 - Step 3 - The function that compute the loss for a data point
def computeLossForASample(s, labelForSample, delta):

    loss_i = 0
    syi = s[
        labelForSample
    ]  # the score for the correct class corresonding to the current input sample based on the label yi
    for idx, sj in enumerate(s):
        if idx != labelForSample:
            loss_i += max(0, sj - syi + delta)

    return loss_i


# TODO - Application 3 - Step 4 - The function that compute the gradient loss for a data point
def computeLossGradientForASample(W, s, currentDataPoint, labelForSample, delta):

    dW_i = np.zeros(W.shape)  # initialize the matrix of gradients with zero
    syi = s[labelForSample]  # establish the score obtained for the true class

    for j, sj in enumerate(s):
        dist = sj - syi + delta

        if j == labelForSample:
            continue

        if dist > 0:
            dW_i[j] = currentDataPoint
            dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint

    return dW_i


def main():

    # Input points in the 4 dimensional space
    # x_train = np.array(
    #     [
    #         [1, 5, 1, 4],
    #         [2, 4, 0, 3],
    #         [2, 1, 3, 3],
    #         [2, 0, 4, 2],
    #         [5, 1, 0, 2],
    #         [4, 2, 1, 1],
    #     ]
    # )

    # # Labels associated with the input points
    # y_train = [0, 0, 1, 1, 2, 2]

    # # Input points for prediction
    # x_test = np.array([[1, 5, 2, 4], [2, 1, 2, 3], [4, 1, 0, 1]])

    # # Labels associated with the testing points
    # y_test = [0, 1, 2]


    x, y = map(np.array, get_iris_data('./Iris.csv'))

    random.shuffle(x)
    random.shuffle(y)

    N = len(x)
    train_limit = int(N * 0.2)

    x_train = x[:train_limit]
    x_test = x[train_limit:]

    y_train = y[:train_limit]
    y_test = y[train_limit:]


    # print(x_train, x_test, y_train, y_test)

    # The matrix of wights
    # W = np.array([[-1, 2, 1, 3], [2, 0, -1, 4], [1, 3, 2, 1]], dtype=np.float)
    # W = np.array(np.random.uniform(size=(150, 4)), dtype=np.float)
    W = np.zeros((150, 4), dtype=np.float)

    delta = 0.1  # margin
    step_size = 1  # weights adjustment ratio

    loss_L = 0
    dW = np.zeros(W.shape)
    epochs = 0
    accuracy = 0

    while accuracy <= 90:
        epochs += 1
        
        # TODO - Application 3 - Step 2 - For each input data...
        for idx, xsample in enumerate(x_train):

            # TODO - Application 3 - Step 2 - ...compute the scores s for all classes (call the method predict)
            s = predict(xsample, W)

            # TODO - Application 3 - Step 3 - Call the function (computeLossForASample) that
            #  compute the loss for a data point (loss_i)
            loss_i = computeLossForASample(s, y_train[idx], delta)

            # Print the scores - Uncomment this
            # print(
            #     "Scores for sample {} with label {} is: {} and loss is {}".format(
            #         idx, y_train[idx], s, loss_i
            #     )
            # )

            # TODO - Application 3 - Step 4 - Call the function (computeLossGradientForASample) that
            #  compute the gradient loss for a data point (dW_i)
            dW_i = computeLossGradientForASample(W, s, x_train[idx], y_train[idx], delta)

            # TODO - Application 3 - Step 5 - Compute the global loss for all the samples (loss_L)
            loss_L += loss_i

            # TODO - Application 3 - Step 6 - Compute the global gradient loss matrix (dW)
            dW += dW_i


        # TODO - Application 3 - Step 7 - Compute the global normalized loss
        loss_L_norm = loss_L / len(y_train)
        # print("The global normalized loss = {}".format(loss_L_norm))

        # TODO - Application 3 - Step 8 - Compute the global normalized gradient loss matrix
        dW = dW / len(y_train)

        # TODO - Application 3 - Step 9 - Adjust the weights matrix
        W = W - step_size * dW


        correctPredicted = 0
        for idx, xsample in enumerate(x_test):
            if np.argmax(predict(xsample, W)) == y_test[idx]:
                correctPredicted += 1

        accuracy = correctPredicted / len(y_test) * 100
        print("Accuracy for test = {}%".format(accuracy))

    return


if __name__ == "__main__":
    main()
