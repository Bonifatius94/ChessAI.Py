
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten

from .fext_base_model import ChessFeatureExtractionModel


class ChessRatingModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessRatingModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_feature_ext = ChessFeatureExtractionModel(params)
        self.nn_feature_ext.trainable = params['is_fx_trainable']
        self.flatten = Flatten()
        self.dropout_1 = Dropout(rate=0.5)
        self.dropout_1.build((None, 256))
        self.nn_dense_1 = Dense(units=512, activation='relu')
        self.nn_dense_2 = Dense(units=128, activation='relu')
        self.nn_dense_out = Dense(units=1)


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