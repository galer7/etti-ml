import cv2
import tensorflow as tf
import keras
from CatsAndDogsClassification import defineCNNModelFromScratch


def load_images(target_size):
    test1 = cv2.imread("./test1.jpg")
    test2 = cv2.imread("./test2.jpg")

    test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2RGB)
    test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2RGB)

    test1 = cv2.resize(test1, target_size)
    test2 = cv2.resize(test2, target_size)

    tensor = tf.constant([test1, test2]) / 255

    return tensor


def main():
    target_size = (64, 64)
    tensor = load_images(target_size)

    weights = "./Model_cats_dogs_small_dataset.h5"

    model = defineCNNModelFromScratch(target_size)
    model.load_weights(weights)

    scores = model.predict(tensor, batch_size=2)
    print(scores)

if __name__ == '__main__':
    main()