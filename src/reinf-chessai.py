
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
import os
import math


# training settings
mutations_count = 1000
training_epoch = 100
parallel_processes = 16
results_out_dir = os.environ['MODEL_OUT']
# TODO: make those settings parameterizable with program args


def main():

    # start the training
    do_reinforcement_learning()


def do_reinforcement_learning():

    # init keras model for first iteration
    # TODO: check if this model fits the problem
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

    # initialize the model optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    # compile the model and save it
    best_estimator.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    epoch = 0
    best_win_rate = 0.0

    # do endless training loop
    # TODO: think of a graceful termination mechanism
    while True:

        # create training sessions
        sessions = [(i, TrainingSession(i, optimizer.get_updates(best_estimator.trainable_weights, 
            best_estimator.constraints, loss(best_win_rate, None)))) for i in range(mutations_count)]
        
        # reinforcement training in parallel
        win_rates_by_id = []
        with multiprocessing.Pool(processes=parallel_processes) as pool:
            win_rates_by_id = pool.map(TrainingSession.do_training, sessions)

        # pick the best estimator (highest win rate)
        sorted_win_rates = [x[1] for x in win_rates_by_id.sort()]
        index = np.argmax(np.array(sorted_win_rates))
        best_estimator = sessions[index].mutated_estimator
        best_win_rate = sorted_win_rates[index]

        # evaluate estimator vs. stockfish engine
        test_win_rate = test_estimator_vs_stockfish(best_estimator)

        # save the estimator to file
        save_model_weights(best_estimator, epoch)

        # print the 
        print("training epoch:", epoch, "training loss:", 
            loss(best_win_rate, None), "test loss:", loss(test_win_rate, None))


def test_estimator_vs_stockfish(best_estimator):

    win_rate = 0.0

    # TODO: implement test logic

    return win_rate


def loss(win_rate, dummy):
    return math.exp(-1 * (win_rate - 0.5))


def save_model(model):

    # serialize model to JSON
    model_json = model.to_json()
    with open("chess-ai-model.json", "w") as json_file:
        json_file.write(model_json)


def save_model_weights(model, epoch):

    # serialize weights to HDF5
    model.save_weights("chess-ai-model-weights-{}.h5".format(epoch))


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