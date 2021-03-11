
import tensorflow as tf
from tensorflow import keras
import numpy as np
#import matplotlib.pyplot as plt


# init global training data sets
train_labels = []
test_labels = []
train_images = []
test_images = []
class_names = []


def main():

    init_training_data()
    do_training()


def init_training_data():

    global train_images
    global train_labels
    global test_images
    global test_labels
    global class_names

    # load fashion dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # init image values (normalize as float value between 0 and 1)
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # init classes to be trained
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def do_training():

    sess = tf.Session()
    with sess.as_default():

        # define neuronal network
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        # compile the network
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # do the training
        model.fit(train_images, train_labels, epochs=10)

        # evaluate the training quality
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

        # get predictions
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = [np.argmax(x) for x in probability_model.predict(test_images)]

        # get confusion matrix evaluating the quality of the predictions
        conf_matrix = tf.math.confusion_matrix(test_labels, predictions)
        print(conf_matrix.eval())


# start main program
main()
