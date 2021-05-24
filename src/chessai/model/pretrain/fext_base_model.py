
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D


class ChessFeatureExtractionModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessFeatureExtractionModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_conv_1 = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')
        self.max_pool_1 = MaxPool2D()
        self.nn_conv_2 = Conv2D(16, (5, 5), strides=1, padding='same', activation='relu')
        self.max_pool_2 = MaxPool2D()
        self.nn_conv_3 = Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')
        self.max_pool_3 = MaxPool2D()
        self.nn_conv_4 = Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')
        self.max_pool_4 = MaxPool2D()
        self.nn_conv_5 = Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')

        # specify whether the feature extraction is supposed to be trainable
        self.trainable(params['is_fx_trainable'])


    def call(self, inputs, training=False):

        x = inputs
        x = self.max_pool_1(self.nn_conv_1(x))
        x = self.max_pool_2(self.nn_conv_2(x))
        x = self.max_pool_3(self.nn_conv_3(x))
        x = self.max_pool_4(self.nn_conv_4(x))
        x = self.nn_conv_5(x)
        return x