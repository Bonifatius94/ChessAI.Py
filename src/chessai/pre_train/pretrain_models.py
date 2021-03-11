
import tensorflow as tf


class ChessFeatureExtractionModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessFeatureExtractionModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_conv_1 = tf.keras.layers.Conv2D(16, (8, 8), strides=1, padding='same', activation='relu')
        self.nn_conv_2 = tf.keras.layers.Conv2D(16, (6, 6), strides=1, padding='same', activation='relu')
        self.nn_conv_3 = tf.keras.layers.Conv2D(16, (5, 5), strides=1, padding='same', activation='relu')
        self.nn_conv_4 = tf.keras.layers.Conv2D(16, (4, 4), strides=1, padding='same', activation='relu')
        self.nn_conv_5 = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')
        self.nn_conv_6 = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')
        self.nn_flatten_output = tf.keras.layers.Flatten()
        self.nn_dense_output = tf.keras.layers.Dense(units=512)


    def call(self, inputs):

        x = inputs
        x = self.nn_conv_1(x)
        x = self.nn_conv_2(x)
        x = self.nn_conv_3(x)
        x = self.nn_conv_4(x)
        x = self.nn_conv_5(x)
        x = self.nn_conv_6(x)
        x = self.nn_flatten_output(x)
        x = self.nn_dense_output(x)
        return x


class ChessDrawGenerator(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessDrawGenerator, self).__init__()
        self.params = params

        # create model input layers
        self.nn_feature_ext = ChessFeatureExtractionModel()
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


    def call(self, chessboard, drawing_pos):

        # initialize LSTM state ('show' to board to the LSTM)
        board_features = self.nn_feature_ext(chessboard)
        zero_state = self.nn_stacked_lstm_cells.get_initial_state(
            batch_size=self.params['batch_size'], dtype=tf.float32)
        x, h = self.nn_stacked_lstm_cells(board_features, zero_state)

        # predict a legal draw for the chess piece at the given field position
        x = self.nn_start_pos_embedding(drawing_pos)
        x, h = self.nn_stacked_lstm_cells(x, h)
        x = self.nn_flatten_output(x)
        target_pos = self.nn_dense_output(x)

        return target_pos


class ChessRatingModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessRatingModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_feature_ext = ChessFeatureExtractionModel(params)
        self.nn_dense_1 = tf.keras.layers.Dense(512)
        self.nn_dense_2 = tf.keras.layers.Dense(256)
        self.nn_dense_3 = tf.keras.layers.Dense(128)
        self.nn_dense_4 = tf.keras.layers.Dense(64)
        self.nn_dense_output = tf.keras.layers.Dense(1)


    def call(self, inputs):

        x = inputs
        x = self.nn_feature_ext(x)
        x = self.nn_dense_1(x)
        x = self.nn_dense_2(x)
        x = self.nn_dense_3(x)
        x = self.nn_dense_4(x)
        x = self.nn_dense_output(x)
        return x