
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.regularizers import L1L2


class ChessFeatureExtractionModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessFeatureExtractionModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_conv_1 = Conv2D(64, (7, 7), strides=1, padding='same', activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_conv_2 = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_conv_3 = Conv2D(16, (5, 5), strides=1, padding='same', activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_conv_4 = Conv2D(16, (3, 3), strides=1, padding='same', activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_conv_5 = Conv2D(16, (3, 3), strides=1, padding='same', activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))


    def call(self, inputs, training=False):

        x = inputs
        x = self.nn_conv_1(x)
        x = self.nn_conv_2(x)
        x = self.nn_conv_3(x)
        x = self.nn_conv_4(x)
        x = self.nn_conv_5(x)
        return x