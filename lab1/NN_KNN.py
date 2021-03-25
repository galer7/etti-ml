import numpy as np
import keras
import cv2
from matplotlib import pyplot as plt

#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
dict_classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def most_frequent(l):
    count_labels_dict = {}

    for item in l:
        if item[1] not in count_labels_dict:
            count_labels_dict[item[1]] = 1
        else:
            count_labels_dict[item[1]] += 1

    return max(count_labels_dict, key=lambda k: count_labels_dict[k])

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 1 - Step 3 - Compute the difference between the current image (img) taken from test dataset
#  with all the images from the train dataset. Return the label for the training image for which is obtained
#  the lowest score
def predictLabelNN(x_train_flatten, y_train, img):

    predicted_label_index = -1
    scoreMin = 100000000

    for idx, imgT in enumerate(x_train_flatten):

        difference = abs(np.subtract(img, imgT))
        difference_L2 = difference ** 2

        score = np.sqrt(np.sum(difference_L2))

        if score < scoreMin:
            predicted_label_index = idx
            scoreMin = score


    return y_train[predicted_label_index]
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 2 - Step 1 - Create a function (predictLabelKNN) that predict the label for a test image based
#  on the dominant class obtained when comparing the current image with the k-NearestNeighbor images from train
def predictLabelKNN(x_train_flatten, y_train, img, k=3, diff='L1'):

    predictedLabel = -1
    predictions = []  # list to save the scores and associated labels as pairs  (score, label)
    score = 0

    for idx, imgT in enumerate(x_train_flatten):

        difference = abs(np.subtract(img, imgT))

        if diff == 'L2':
            difference = difference ** 2

        if diff == 'L2':
            score = np.sqrt(np.sum(difference))
        else:
            score = np.sum(difference)

        predictions.append([score, y_train[idx][0]])

    predictions = sorted(predictions, key=lambda x: x[0])

    top_k_predictions = predictions[:k]
    print(top_k_predictions)


    predictedLabel = most_frequent(top_k_predictions)

    return predictedLabel

def do_every_combination(N, x_train_flatten, y_train, x_test_flatten, y_test, diff_type='L1'):
    info = {}

    for i in [1, 3, 5, 10, 20, 50]:
        info[i] = {}

        info[i]['correct_predicted'] = 0
        info[i]['last_predicted'] = 0
        info[i]['accuracy'] = 0

    for idx, img in enumerate(x_test_flatten[0:N]):

        print(f"Make a prediction for image {idx}")


        # TODO - Application 1 - Step 3 - Call the predictLabelNN function
        for key in info:
            info[key]['last_predicted'] = predictLabelKNN(x_train_flatten, y_train, img, k=key, diff=diff_type)



        # TODO - Application 1 - Step 4 - Compare the predicted label with the groundtruth (the label from y_test).
        #  If there is a match then increment the contor numberOfCorrectPredictedImages

        for key in info:
            if info[key]['last_predicted'] == y_test[idx]:
                info[key]['correct_predicted'] += 1


    with open(f"./results_{diff_type}.txt", "w") as f:
        for key in info:
            info[key]['accuracy'] = N * info[key]['correct_predicted'] / 100
            f.write(f"{diff_type} - k={key} ========= {info[key]['accuracy']}\n")
    
    return info

def exercises_4_5(N, diff_type='L1'):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    x_train_flatten = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_test_flatten = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

    do_every_combination(N, x_train_flatten, y_train, x_test_flatten, y_test, diff_type=diff_type)


def main():

    # TODO - Application 1 - Step 1 - Load the CIFAR-10 dataset

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # TODO - Exercise 1 - Determine the size of the four vectors x_train, y_train, x_test, y_test

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # TODO - Exercise 2 - Visualize the first 10 images from the testing dataset with the associated labels

    for i in range(10):
        # define subplot
        ax = plt.subplot(2, 5, 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        ax.set_title(dict_classes[y_train[i][0]])

    plt.show()

    # TODO - Application 1 - Step 2 - Reshape the training and testing dataset from 32x32x3 to a vector
    x_train_flatten = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_test_flatten = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

    numberOfCorrectPredictedImages = 0
    N = 100

    

    # TODO - Application 1 - Step 3 - Predict the labels for the first 100 images existent in the test dataset

    for idx, img in enumerate(x_test_flatten[0:N]):

        print(f"Make a prediction for image {idx}")


        # TODO - Application 1 - Step 3 - Call the predictLabelNN function

        predictLabel = predictLabelNN(x_train_flatten, y_train, img)


        # TODO - Application 1 - Step 4 - Compare the predicted label with the groundtruth (the label from y_test).
        #  If there is a match then increment the contor numberOfCorrectPredictedImages

        if predictLabel == y_test[idx]:
            numberOfCorrectPredictedImages += 1


        # TODO - Application 1 - Step 5 - Compute the accuracy
        accuracy = 100 * numberOfCorrectPredictedImages / N
        print(f"accuracy: {accuracy}%")



if __name__ == '__main__':
    # main()
    exercises_4_5(100, 'L2')

