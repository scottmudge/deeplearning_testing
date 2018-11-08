import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
import cv2 as cv
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import utils.logger
import utils.utility as utility
from utils.logger import get_logger as logging

import numpy as np

if __name__ == '__main__':

    utils.logger.initialize("DeepLearning_Test1")
    model_fn = "{}/model.dat".format(utility.get_root_directory())

    np.random.seed(123)

    # -----------------------------
    # Initialize TensorFlow Session
    # -----------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf.Session(config=config)
    set_session(sess)

    # -----------------------------
    # Initialize Data
    # -----------------------------

    logging("Data").info("Initializing Data...")
    # Load pre-shuffled MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows = 28
    img_cols = 28

    # Should print 60000, 28, 28
    logging("Data").info("x_train shape: {}".format(x_train.shape))

    logging("Data").info("Reshaping...")
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    logging("Data").info("Converting to float...")
    # Convert to float 32 for faster training
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    logging("Data").info("Normalizing...")
    # Normalize data
    x_train /= 255.0
    x_test /= 255.0

    logging("Data").info("y_train shape: {}".format(y_train.shape))
    logging("Data").info("Converting to categorical data...")
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    logging("Data").info("Done, new y_train shape: {}".format(y_train.shape))

    # -----------------------------
    # Define Model Architecture
    # -----------------------------

    logging("Model").info("Creating model...")
    model = Sequential()

    # 32 x 3 x 3 Conv layer with:
    #   > relu activation
    #   > input shape of 1 channel by 28 x 28 pixels
    model.add(Conv2D(32, 3, 3, activation='relu', input_shape=input_shape))

    # Second Conv Layer
    model.add(Conv2D(32, 3, 3, activation='relu'))
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout to prevent over-fitting
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    logging("Model").info("Model shape: {}".format(model.output_shape))

    # Test if model weights already exists
    if utility.file_exists(model_fn):
        logging("Model").info("Loading model weights from disk:\n\t> Filename: {}".format(model_fn))
        model = model.load_weights(model_fn)

    logging("Main").info("Done Loading!\n")
    input("Press Enter to continue...")

    logging("Model").info("Fitting model to data...")
    model.fit(x_train, y_train, batch_size=32, nb_epoch=10, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    logging("Score").info("Final score: \n\t> Loss: {}\n\t> Accuracy: {}".format(score[0], score[1]))


    logging("Model").info("Saving model...")
    model.save(model_fn, True, True)




