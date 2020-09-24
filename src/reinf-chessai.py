
# neuronal network:
# =================
# input: 14 x 64-bit integer array
# some layers with 128 neurons each
# maybe sigmoid activation function (still needs to be tested)
# output: score 0.0 - 1.0

# algorithm:
# ==========
# cache: currently best estimator function (one nn instance)
# init best estimator function randomly
# n mutations of best estimator function
# each mutation plays against the currently best estimator for several times
# after that the best mutated function becomes the new best estimation function

# play game:
# ==========
# choose sides
# until the game is over (do-while):
#   compute all possible draws
#   evaluate the scores of the draws with the est. func. of the drawing side's player
#   pick the 'best' draw and apply it to the board

# rewarding / punishment:
# =======================
# game won  -> +1
# game tied -> +0
# game lost -> -1

# evaluation phase:
# =================
# play against well-known strong chess engines like Stockfish
# -> determine if the learned behavior is actually successful


import tensorflow as tf
from tensorflow import keras
import chesslib
import random
import numpy as np
import multiprocessing


# training settings
mutations_count = 1000
training_epoch = 100


def main():

    # start the training
    do_reinforcement_learning()


def do_reinforcement_learning():

    # init keras model for first iteration
    best_estimator = keras.Sequential([
        keras.layers.Flatten(input_shape=(14,)),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dense(1, activation='linear')
    ])

    # compile the network
    best_estimator.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        # TODO: replace this function with the win rate
    )

    # do endless training loop
    while True:

        # create a training sessions
        sessions = [TrainingSession(i, best_estimator) for i in range(mutations_count)]
        win_rates_by_id = []

        # reinforcement training in parallel
        with multiprocessing.Pool(processes=16) as pool:
            win_rates_by_id = pool.map(TrainingSession.do_training, sessions)

        # pick the best estimator (highest win rate)
        sorted_win_rates = [x[1] for x in win_rates_by_id.sort()]
        index = np.argmax(np.array(sorted_win_rates))
        best_estimator = sessions[index].mutated_estimator

        # TODO: evaluation vs. stockfish engine


class TrainingSession(object):

    def __init__(self, id, best_estimator):

        self.id = id

        # init estimators
        self.best_estimator = best_estimator
        self.mutated_estimator = keras.model.clone_model(best_estimator)
        # TODO: add mutation formula!!!!!! (otherwise there is no learning progess ...)

        # init training variables
        self.wins = 0
        self.ties = 0
        self.defeats = 0

        # init chess game variables
        self.board = chesslib.ChessBoard_StartFormation()
        self.drawing_side = 0
        self.draw_history = []


    def do_training(self):

        # play several training games
        for i in range(training_epoch):
            self.play_chessgame()

        # determine the win rate
        win_rate = 1.0 * self.wins / (self.wins + self.defeats)
        return (id, win_rate)


    def play_chessgame(self):

        # reset it chess game variables
        self.board = chesslib.ChessBoard_StartFormation()
        self.drawing_side = 0
        self.draw_history = []

        # determine side selection
        training_side = random.randint(0, 1)

        # play until the game is over
        while True:

            # compute all possible draws
            last_draw = self.draw_history[-1] if len(self.draw_history) > 0 else chesslib.ChessDraw_Null
            draws = chesslib.GenerateDraws(self.board, training_side, last_draw, True)
            vector = np.append(draws, np.full(len(draws), last_draw))

            # determine the best of those draws using the estimator
            model = self.mutated_estimator if self.drawing_side == training_side else self.best_estimator
            predictions = model.predict(vector)
            best_draw = draws[np.argmax(predictions)]
            # TODO: check if this does the right stuff

            # apply the draw to the chessboard
            self.board = chesslib.ApplyDraw(self.board, best_draw)

            # update game variables
            self.draw_history.append(best_draw)

            # exit if the game is over
            if self.is_game_over():
                break


    def is_game_over(self):

        # don't compute this function if the board is still in start formation
        if len(self.draw_history) == 0:
            return False

        # determine whether the game is over
        step_draw = self.draw_history[-2] if len(self.draw_history) >= 2 else chesslib.ChessDraw_Null
        last_draw = self.draw_history[-1] if len(self.draw_history) >= 1 else chesslib.ChessDraw_Null
        state = chesslib.GameState(self.board, last_draw)
        enemy_draws = chesslib.GenerateDraws(self.board, self.drawing_side, step_draw, True)

        game_over = True

        # update the game's outcome stats if the game is over (win / loss / tie)
        if state != chesslib.GameSate_Checkmate:
            self.defeats += 1
        elif state != chesslib.GameState_Tie:
            self.ties += 1
        elif len(enemy_draws) == 0:
            self.wins += 1
        # otherwise just continue (game is not over yet)
        else:
            game_over = False

        return game_over


# start main function
main()