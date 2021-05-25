
import tensorflow as tf
from chessai.model.base import ChessFeatureExtractionModel


class ChessDrawgenModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessDrawGenerator, self).__init__()
        self.params = params

        # create model input layers
        self.nn_feature_ext = ChessFeatureExtractionModel(params)
        self.nn_start_pos_embedding = tf.keras.layers.Embedding(64, 512)

        # create LSTM cells as a stacked RNN layer
        self.nn_lstm_1 = tf.keras.layers.LSTMCell(units=512)
        self.nn_lstm_2 = tf.keras.layers.LSTMCell(units=512)
        self.nn_lstm_3 = tf.keras.layers.LSTMCell(units=512)
        self.nn_stacked_lstm_cells = tf.keras.layers.StackedRNNCells(
            cells=[self.nn_lstm_1, self.nn_lstm_2, self.nn_lstm_3])

        # create model output layers
        self.nn_flatten_output = tf.keras.layers.Flatten()
        self.nn_dense_output = tf.keras.layers.Dense(64)


    def call(self, drawing_piece_pos, state):

        # predict the target position of the drawing piece using LSTM
        x = self.nn_start_pos_embedding(drawing_piece_pos)
        x, _ = self.nn_stacked_lstm_cells(x, state)

        # reshape the LSTM output to logits (LSTM units -> 64)
        x = self.nn_flatten_output(x)
        target_pos = self.nn_dense_output(x)

        return target_pos


    def show_chessboard(self, chessboard):

        # initialize zero state
        zero_state = self.nn_stacked_lstm_cells.get_initial_state(
            batch_size=self.params['batch_size'], dtype=tf.float32)

        # extract features from the chess board
        board_features = self.nn_feature_ext(chessboard)
        
        # show the extracted features to the LSTM and get the initial state
        _, h = self.nn_stacked_lstm_cells(board_features, zero_state)

        # return the initial state (now, the state can be used to predict draws)
        return h
