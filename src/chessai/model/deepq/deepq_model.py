
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.regularizers import L1L2

from chessai.model.base import ChessFeatureExtractionModel


class ChessDeepQModel(tf.keras.Model):

    def __init__(self, dropout_rate: float=0.2, is_fx_trainable: bool=True,
                 l1_penalty: float=1e-4, l2_penalty: float=1e-5):

        # call super constructor and store overloaded parameters
        super(ChessDeepQModel, self).__init__()

        # create feature extraction layer
        self.nn_feature_ext = ChessFeatureExtractionModel(None)
        self.nn_feature_ext.trainable = is_fx_trainable

        # create layers to convert the fx content
        self.flatten = Flatten()
        self.dropout_1 = Dropout(rate=dropout_rate)

        # create fully-conn layers rating the extracted features
        self.nn_dense_1 = Dense(units=512, activation='relu',
                                kernel_regularizer=L1L2(l1=l1_penalty, l2=l2_penalty))
        self.nn_dense_2 = Dense(units=128, activation='relu',
                                kernel_regularizer=L1L2(l1=l1_penalty, l2=l2_penalty))
        self.nn_dense_out = Dense(units=1, activation='sigmoid',
                                  kernel_regularizer=L1L2(l1=l1_penalty, l2=l2_penalty))
        # info: the sigmoid activation on the last layer produces output values within [-1, 1]


    def call(self, inputs, training=False):

        x = inputs

        # extract the features from chess bitboards
        x = self.flatten(self.nn_feature_ext(x))
        x = self.dropout_1(x, training)

        # fuse the features to a single score value
        x = self.nn_dense_1(x)
        x = self.nn_dense_2(x)
        x = self.nn_dense_out(x)

        return x