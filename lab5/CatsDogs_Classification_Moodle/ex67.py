from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, MaxPooling2D
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.applications import VGG16

import os
import shutil


def prepareDatabase(original_directory, base_directory):

    # If the folder already exist remove everything
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)

    # Recreate the basefolder
    os.mkdir(base_directory)

    # TODO - Application 1 - Step 1 - Struncture the dataset in training, validation and testing directories

    train_directory = os.path.join(base_directory, "train")
    os.mkdir(train_directory)
    validation_directory = os.path.join(base_directory, "validation")
    os.mkdir(validation_directory)
    test_directory = os.path.join(base_directory, "test")
    os.mkdir(test_directory)

    # TODO - Application 1 - Step 1 - Create the cat/dog training directories - See figure 4

    # create the cat/dog training directories
    train_cats_directory = os.path.join(train_directory, "cats")
    os.mkdir(train_cats_directory)
    train_dogs_directory = os.path.join(train_directory, "dogs")
    os.mkdir(train_dogs_directory)

    # TODO - Application 1 - Step 1 - Create the cat/dog validation directories - See figure 4

    # create the cat/dog validation directories
    validation_cats_directory = os.path.join(validation_directory, "cats")
    os.mkdir(validation_cats_directory)
    validation_dogs_directory = os.path.join(validation_directory, "dogs")
    os.mkdir(validation_dogs_directory)

    # TODO - Application 1 - Step 1 - Create the cat/dog testing directories - See figure 4

    # create the cat/dog testing directories
    test_cats_directory = os.path.join(test_directory, "cats")
    os.mkdir(test_cats_directory)
    test_dogs_directory = os.path.join(test_directory, "dogs")
    os.mkdir(test_dogs_directory)

    # TODO - Application 1 - Step 1 - Copy the first 1000 cat images in to the training directory (train_cats_directory)

    # copy the first 1000 cat images in to the training directory (train_cats_directory)
    original_directory_cat = os.path.join(original_directory, "cats")
    fnames = ["{}.jpg".format(i) for i in range(1000)]

    for fname in fnames:
        src = os.path.join(original_directory_cat, fname)
        dst = os.path.join(train_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 1 - Copy the next 500 cat images in to the validation directory (validation_cats_directory)

    fnames = ["{}.jpg".format(i) for i in range(1000, 1500)]

    for fname in fnames:
        src = os.path.join(original_directory_cat, fname)
        dst = os.path.join(validation_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 1 - Copy the next 500 cat images in to the test directory (test_cats_directory)

    fnames = ["{}.jpg".format(i) for i in range(1500, 2000)]

    for fname in fnames:
        src = os.path.join(original_directory_cat, fname)
        dst = os.path.join(test_cats_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 2 - Copy the first 1000 dogs images in to the training directory (train_dogs_directory)

    original_directory_dog = os.path.join(original_directory, "dogs")
    fnames = ["{}.jpg".format(i) for i in range(1000)]

    for fname in fnames:
        src = os.path.join(original_directory_dog, fname)
        dst = os.path.join(train_dogs_directory, fname)
        shutil.copyfile(src, dst)

        # TODO - Application 1 - Exercise 2 - Copy the next 500 dogs images in to the validation directory (validation_dogs_directory)

        fnames = ["{}.jpg".format(i) for i in range(1000, 1500)]

    for fname in fnames:
        src = os.path.join(original_directory_dog, fname)
        dst = os.path.join(validation_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Exercise 2 - Copy the next 500 dogs images in to the test directory (test_dogs_directory)

    fnames = ["{}.jpg".format(i) for i in range(1500, 2000)]

    for fname in fnames:
        src = os.path.join(original_directory_dog, fname)
        dst = os.path.join(test_dogs_directory, fname)
        shutil.copyfile(src, dst)

    # TODO - Application 1 - Step 1 - As a sanitary check verify how many pictures are in each directory

    # As a sanitary check verify how many pictures are in each directory
    print(
        "Total number of CATS used for training = {}".format(
            len(os.listdir(train_cats_directory))
        )
    )
    print(
        "Total number of CATS used for validation = {}".format(
            len(os.listdir(validation_cats_directory))
        )
    )
    print(
        "Total number of CATS used for testing = {}".format(
            len(os.listdir(test_cats_directory))
        )
    )
    print(
        "Total number of DOGS used for training = {}".format(
            len(os.listdir(train_dogs_directory))
        )
    )
    print(
        "Total number of DOGS used for validation = {}".format(
            len(os.listdir(validation_dogs_directory))
        )
    )
    print(
        "Total number of DOGS used for testing = {}".format(
            len(os.listdir(test_dogs_directory))
        )
    )

    return


def defineCNNModelFromScratch(target_size):

    # Application 1 - Step 3 - Initialize the sequential model
    model = models.Sequential()

    # TODO - Application 1 - Step 3 - Create the first hidden layer as a convolutional layer
    model.add(Conv2D(32, 3, input_shape=(*target_size, 3), activation="relu"))

    # TODO - Application 1 - Step 3 - Define a pooling layer
    model.add(MaxPooling2D())

    # TODO - Application 1 - Step 3 - Create the third hidden layer as a convolutional layer
    model.add(Conv2D(64, 3))

    # TODO - Application 1 - Step 3 - Define a pooling layer
    model.add(MaxPooling2D())

    # TODO - Application 1 - Step 3 - Create another convolutional layer
    model.add(Conv2D(128, 3))

    # TODO - Application 1 - Step 3 - Define a pooling layer
    model.add(MaxPooling2D())

    # TODO - Application 1 - Step 3 - Create another convolutional layer
    model.add(Conv2D(128, 3))

    # TODO - Application 1 - Step 3 - Define a pooling layer
    model.add(MaxPooling2D())

    # TODO - Application 1 - Step 3 - Define the flatten layer
    model.add(Flatten())

    # TODO - Application 1 - Step 3 - Define a dense layer of size 512
    model.add(Dense(512))

    # TODO - Application 1 - Step 3 - Define the output layer
    model.add(Dense(1, activation="sigmoid"))

    # TODO - Application 1 - Step 3 - Visualize the network arhitecture (list of layers)
    model.summary()

    # TODO - Application 1 - Step 3 - Compile the model
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def defineCNNModelVGGPretrained(target_size):

    # TODO - Exercise 6 - Load the pretrained VGG16 network in a variable called base_model
    # The top layers will be omitted; The input_shape will be kept to (150, 150, 3)
    base_model = VGG16(include_top=False, input_shape=(*target_size, 3))

    # TODO - Exercise 6 - Visualize the network arhitecture (list of layers)
    base_model.summary()

    # TODO - Exercise 6 - Freeze the base_model layers to not to allow training
    for layer in base_model.layers:
        if "block5_conv" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    # Create the final model and add the layers from the base_model
    VGG_model = models.Sequential()
    VGG_model.add(base_model)

    # TODO - Exercise 6 - Add the flatten layer
    VGG_model.add(Flatten())

    # TODO - Exercise 6 - Add the dropout layer
    VGG_model.add(Dropout(0.5))

    # TODO - Exercise 6 - Add a dense layer of size 512
    VGG_model.add(Dense(512, activation="relu"))

    # TODO - Exercise 6 - Add the output layer
    VGG_model.add(Dense(1, activation="sigmoid"))

    VGG_model.summary()

    # TODO - Exercise 6 - Compile the model
    VGG_model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return VGG_model


def imagePreprocessing(base_directory, target_size):

    train_directory = base_directory + "/train"
    validation_directory = base_directory + "/validation"

    # TODO - Application 1 - Step 2 - Create the image data generators for train and validation
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # TODO - Application 1 - Step 2 - Analize the output of the train and validation generators
    train_generator = train_datagen.flow_from_directory(
        train_directory, target_size=target_size, batch_size=20, class_mode="binary"
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=target_size,
        batch_size=20,
        class_mode="binary",
    )

    return train_generator, validation_generator


def visualizeTheTrainingPerformances(history):

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    
    pyplot.figure()
    pyplot.title("Training and validation accuracy")
    pyplot.plot(epochs, acc, "bo", label="Training accuracy")
    pyplot.plot(epochs, val_acc, "b", label="Validation accuracy")
    pyplot.legend()

    pyplot.figure()
    pyplot.title("Training and validation loss")
    pyplot.plot(epochs, loss, "bo", label="Training loss")
    pyplot.plot(epochs, val_loss, "b", label="Validation loss")
    pyplot.legend()

    pyplot.show()

    return


def main():

    original_directory = "../Kaggle_Cats_And_Dogs_Dataset"
    base_directory = "Kaggle_Cats_And_Dogs_Dataset_Small"
    target_size = (64, 64)

    prepareDatabase(original_directory, base_directory)

    # TODO - Application 1 - Step 2 - Call the imagePreprocessing method
    train_generator, validation_generator = imagePreprocessing(
        base_directory, target_size
    )

    # TODO - Application 1 - Step 3 - Call the method that creates the CNN model
    model = defineCNNModelVGGPretrained(target_size)

    # TODO - Application 1 - Step 4 - Train the model

    weights_name = "Model_cats_dogs_small_dataset_VGG16.h5"

    if not os.path.exists(weights_name):
        history = model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=50,
        )
        model.save(weights_name)
        # TODO - Application 1 - Step 5 - Visualize the system performance using the diagnostic curves
        visualizeTheTrainingPerformances(history)
    else:
        model.load_weights(weights_name)


    scores = model.evaluate(validation_generator)

    return


if __name__ == "__main__":
    main()
