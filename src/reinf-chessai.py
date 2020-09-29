
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
import tensorflow.keras.backend as K
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
learning_rate = 0.01
results_out_dir = os.environ['MODEL_OUT']
# TODO: make those settings parameterizable with program args


def main():

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

    # compile the model and save it
    #best_estimator.compile(optimizer="Adam")

    epoch = 0
    best_win_rate = 0.0

    # do endless training loop
    # TODO: think of a graceful termination mechanism
    for k in range(1000):

        # create training sessions
        mutations = [mutate_model(best_estimator, learning_rate) for i in range(mutations_count)]
        sessions = [(i, TrainingSession(i, best_estimator, mutations[i])) for i in range(mutations_count)]

        win_rates_by_id = []
        for sess in sessions:
            win_rate = sess.do_training()
            win_rates_by_id.append((id, win_rate))

        # reinforcement training in parallel
        #win_rates_by_id = []
        #with multiprocessing.Pool(processes=parallel_processes) as pool:
        #    win_rates_by_id = pool.map(TrainingSession.do_training, sessions)

        # pick the best estimator (highest win rate)
        sorted_win_rates = [x[1] for x in win_rates_by_id.sort()]
        index = np.argmax(np.array(sorted_win_rates))
        new_best_win_rate = sorted_win_rates[index]

        # only override old estimator if there is an improvement
        if new_best_win_rate > best_win_rate:
            best_estimator = sessions[index].mutated_estimator
            best_win_rate = new_best_win_rate

        # evaluate estimator vs. stockfish engine
        test_win_rate = test_estimator_vs_stockfish(best_estimator)

        # save the estimator to file
        save_model_weights(best_estimator, epoch)

        # print the training progress
        print("training epoch:", epoch, "training loss:", best_win_rate, "test loss:", test_win_rate)


def test_estimator_vs_stockfish(best_estimator):

    win_rate = 0.0

    # TODO: implement test logic

    return win_rate


def mutate_model(orig_model: keras.Model, learning_rate):

    # clone the given model
    model = tf.keras.models.clone_model(orig_model)

    # get the model's weights
    weights = model.get_weights()

    print("updating weights")
    print(len(weights))

    for i in range(len(weights)):

        # calculate the differential weight changes (random uniform distribution)
        shape = weights[i].shape
        updates = np.random.uniform(-learning_rate, learning_rate, shape)

        old_weights = weights[i]
        updated_weights = np.add(weights[i], updates)

        upper_bound = updated_weights >= -1
        lower_bound = updated_weights <= 1
        within_range = np.logical_and(upper_bound, lower_bound)
        outside_range = np.logical_not(within_range)

        if len(old_weights[outside_range]) > 0:
            updated_weights = np.add(updated_weights[within_range], old_weights[outside_range])

        # apply updates to the weights
        weights[i] = updated_weights

    # update the model with the new weights and return the updated model
    model.set_weights(weights)

    return model


def save_model(model):

    # serialize model to JSON
    model_json = model.to_json()
    with open("chess-ai-model.json", "w") as json_file:
        json_file.write(model_json)


def save_model_weights(model, epoch):

    # serialize weights to HDF5
    out_file_path = "chess-ai-model-weights-{}.h5".format(epoch)
    model.save_weights(out_file_path)


class TrainingSession(object):

    def __init__(self, id, best_estimator, mutated_estimator):

        self.id = id

        # init estimators
        self.best_estimator = best_estimator
        self.mutated_estimator = mutated_estimator
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

        print("start training session #", self.id)

        # play several training games
        for i in range(training_epoch):
            self.play_chessgame()

        # determine the win rate
        win_rate = 1.0 * self.wins / (self.wins + self.defeats)

        print("end training session #", self.id)
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