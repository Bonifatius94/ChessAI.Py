
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.regularizers import L1L2

from chessai.model.base import ChessFeatureExtractionModel


class OnesideChessRatingModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(OnesideChessRatingModel, self).__init__()
        self.params = params

        # create model layers
        self.input = Input(shape=(8, 8, 13))
        self.nn_feature_ext = ChessFeatureExtractionModel(params)
        self.nn_feature_ext.trainable = params['is_fx_trainable']
        self.flatten = Flatten()
        self.dropout_1 = Dropout(rate=params['dropout_rate'])

        #self.nn_dense_1 = Dense(units=512, activation='relu')
        #                         kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_dense_2 = Dense(units=128, activation='relu',
                                kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))
        self.nn_dense_out = Dense(units=1, activation='sigmoid')
                                  #kernel_regularizer=L1L2(l1=params['l1_penalty'], l2=params['l2_penalty']))


    def call(self, inputs, training=False):

        x = self.input(inputs)

        # extract the features from chess bitboards
        x = self.flatten(self.nn_feature_ext(x))
        self.dropout_1(x, training)

        # fuse the features to a single score value
        #x = self.nn_dense_1(x)
        x = self.nn_dense_2(x)
        x = self.nn_dense_out(x)

        return x


class ChessRatingModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessRatingModel, self).__init__()
        self.params = params

        # create a model for either side
        self.white_model = OnesideChessRatingModel(params)
        self.black_model = OnesideChessRatingModel(params)


    def call(self, inputs, training: bool=False):

        # unwrap inputs
        board, side = inputs

        # predict for either side (side: 0=white, 1=black)
        pred_white = (1 - side) * self.white_model(board, training)
        pred_black = side * self.black_model(board, training)
        return pred_white + pred_black
        # TODO: check if tensorflow is smart enough not to evaluate the prediction that is 0

        # return self.white_model(board, training) if side == 0 \
        #     else self.white_model(board, training)