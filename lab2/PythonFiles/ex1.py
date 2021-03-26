import numpy as np


def activationFunction(n):

    # TODO - Application 1 - Step 4b - Define the binary step function as activation function

    return 1 if n >= 0 else 0


def forwardPropagation(p, weights, bias):

    a = None  # the neuron output

    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n
    res = np.dot(p, weights) + bias

    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a
    a = activationFunction(res)

    return a


def main():

    # Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    # The network should receive as input two values (0 or 1) and should predict the target output

    # Input data
    P = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Labels
    t = [0, 0, 0, 1]

    # TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    w = [0, 0]

    # TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    b = 0

    # TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    epochs = 0

    # TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    while w != [2, 1] or b != -3:
        epochs += 1

        for i in range(len(t)):

            # TODO - Application 1 - Step 4 - Call the forwardPropagation method
            prediction = forwardPropagation(P[i], w, b)

            # TODO - Application 5 - Compute the prediction error (error)
            err = t[i] - prediction

            # TODO - Application 6 - Update the weights
            w[0] += err * P[i][0]
            w[1] += err * P[i][1]

            # should work?
            # w += err * P

            # TODO - Update the bias
            b += err

    # TODO - Application 1 - Print weights and bias
    print(f"weights: {w}, bias: {b}")
    print(f"epochs needed: {epochs}")

    # TODO - Application 1 - Step 7 - Display the results

    return


if __name__ == "__main__":
    main()
