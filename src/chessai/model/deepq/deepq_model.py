
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.regularizers import L1L2

from chessai.model.base import ChessFeatureExtractionModel


class ChessDeepQModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessDeepQModel, self).__init__()
        self.params = params

        # create feature extraction layer
        self.nn_feature_ext = ChessFeatureExtractionModel(params)
        self.nn_feature_ext.trainable = params['is_fx_trainable']

        # create layers to convert the fx content
        self.flatten = Flatten()
        self.dropout_1 = Dropout(rate=params['dropout_rate'])
        self.dropout_1.build((None, 1024))

        # create fully-conn layers rating the extracted features
        self.nn_dense_1 = Dense(units=512, activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_dense_2 = Dense(units=128, activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_dense_out = Dense(units=1, activation='sigmoid',
                                  kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        # info: the sigmoid activation on the last layer produces output values within [-1, 1]


    def call(self, inputs, training=False):

        x = inputs

        # extract the features from chess bitboards
        x = self.flatten(self.nn_feature_ext(x))
        if training: x = self.dropout_1(x)

        # fuse the features to a single score value
        x = self.nn_dense_1(x)
        x = self.nn_dense_2(x)
        x = self.nn_dense_out(x)

        return x