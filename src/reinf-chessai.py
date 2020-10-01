
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

import os
import tensorflow as tf
from tensorflow import keras
import chesslib
import random
import numpy as np
import math


# configure tensorflow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# training settings
mutations_count = 1000
training_epoch = 100
parallel_processes = 16
learning_rate = 0.01
results_out_dir = os.environ['MODEL_OUT']
# TODO: make those settings parameterizable with program args


def main():

    print("starting chess reinforcement learning")

    # init keras model for first iteration
    # TODO: check if this model fits the problem
    best_estimator = keras.Sequential([
        keras.layers.Flatten(input_shape=(14,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])

    epoch = 0

    # load 'best' model weights if weights are saved
    if os.path.isdir(results_out_dir):
        files = os.listdir(results_out_dir)
        epoch = -1
        for file in files:
            end_index = file.find('.')
            epoch = max([epoch, int(file[23:end_index])])
        if epoch > -1:
            print("loading existing model from epoch ", epoch)
            load_model_weights(best_estimator, epoch)

    epoch += 1

    # do endless training loop
    while True:

        print("starting epoch", epoch)

        # execute training session
        mutated_estimator = mutate_model(best_estimator, learning_rate)
        win_rate = do_training(best_estimator, mutated_estimator)

        # only override old estimator if there is an improvement
        if win_rate > 0.5:

            # update 'best' cache
            best_estimator = mutated_estimator

            # evaluate estimator vs. stockfish engine
            test_win_rate = test_estimator_vs_stockfish(best_estimator)

            # save the estimator to file
            save_model_weights(best_estimator, epoch)

            # print the training progress
            print("training epoch:", epoch, "training loss:", (1 - win_rate), 
                "test loss:", (1 - test_win_rate))

        elif win_rate == 0:

            # update 'best' cache to gain distance from unsuccessful model
            best_estimator = mutated_estimator

        epoch += 1


def test_estimator_vs_stockfish(best_estimator: keras.Model):
    # TODO: implement test logic
    return 0.0


def mutate_model(orig_model: keras.Model, learning_rate: float):

    # clone the given model
    model = tf.keras.models.clone_model(orig_model)

    # get the model's weights
    weights = model.get_weights()

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


def save_model(model: keras.Model):

    # serialize model to JSON
    model_json = model.to_json()
    file_path = os.path.join(results_out_dir, "chess-ai-model.json")
    with open(file_path, "w") as json_file:
        json_file.write(model_json)


def save_model_weights(model: keras.Model, epoch: int):

    # serialize weights to HDF5
    file_name = "chess-ai-model-weights-{}.h5".format(epoch)
    file_path = os.path.join(results_out_dir, file_name)
    model.save_weights(file_path)


def load_model_weights(model: keras.Model, epoch: int):

    # serialize weights to HDF5
    file_name = "chess-ai-model-weights-{}.h5".format(epoch)
    file_path = os.path.join(results_out_dir, file_name)
    model.load_weights(file_path)


def do_training(best_estimator: keras.Model, mutated_estimator: keras.Model):

    wins = 0
    ties = 0
    defeats = 0

    # play one game each on white and black side
    for training_side in range(2):

        # play one game
        game_state = play_chessgame(best_estimator, mutated_estimator, training_side)

        # apply the game's outcome
        if game_state == 'w':
            wins += 1
        elif game_state == 'd':
            defeats += 1
        elif game_state == 't':
            ties += 1

    # determine the win rate
    win_rate = 1.0 * wins / (wins + defeats) if (wins + defeats) > 0 else 0.0
    return win_rate


def play_chessgame(best_estimator: keras.Model, mutated_estimator: keras.Model, training_side):

    # reset it chess game variables
    board = chesslib.ChessBoard_StartFormation()
    drawing_side = 0
    draw_history = []
    game_state = 'n'
    round = 1.0

    print(chesslib.VisualizeBoard(board))

    # play until the game is over
    while True:

        # compute all possible draws
        last_draw = draw_history[-1] if len(draw_history) > 0 else chesslib.ChessDraw_Null
        draws = chesslib.GenerateDraws(board, drawing_side, last_draw, True)
        possible_boards = np.array([chesslib.ApplyDraw(board, draw) for draw in draws])
        fill_column = np.expand_dims(np.full(len(draws), last_draw), axis=1)
        vector = np.append(possible_boards, fill_column, axis=1)

        # determine the best of those draws using the estimator
        model = mutated_estimator if drawing_side == training_side else best_estimator
        predictions = model.predict(vector)
        best_draw = draws[np.argmax(predictions)]

        # apply the draw to the chessboard and update the draw history
        board = chesslib.ApplyDraw(board, best_draw)
        draw_history.append(best_draw)

        # print the board
        print(chesslib.VisualizeDraw(best_draw))
        print(chesslib.VisualizeBoard(board))

        # exit if the game is over
        game_state = get_game_state(board, draw_history, drawing_side)
        if game_state != 'n':
            break

        if has_loops(draw_history):
            game_state = 't'
            break

        drawing_side = (drawing_side + 1) % 2
        round += 0.5

    return game_state


def get_game_state(board: np.array, draw_history: list, drawing_side: int):

    # don't compute this function if the board is still in start formation
    if len(draw_history) == 0:
        return 'n'

    # determine whether the game is over
    step_draw = draw_history[-2] if len(draw_history) >= 2 else chesslib.ChessDraw_Null
    last_draw = draw_history[-1] if len(draw_history) >= 1 else chesslib.ChessDraw_Null
    state = chesslib.GameState(board, last_draw)
    enemy_draws = chesslib.GenerateDraws(board, drawing_side, step_draw, True)

    # determine the game's outcome (not-decided / win / loss / tie)
    game_state = 'n'
    if state == chesslib.GameState_Checkmate:
        game_state = 'd'
    elif state == chesslib.GameState_Tie:
        game_state = 't'
    elif len(enemy_draws) == 0:
        game_state = 'w'

    return game_state


def has_loops(draw_history: list):

    loop_size = 4
    draws_len = len(draw_history)

    # loop through search size
    while loop_size <= math.floor(draws_len / 2):

        rest_draws = draw_history[:-loop_size]
        loop_draws = draw_history[-loop_size:]

        # loop through offsets
        for diff in range(len(rest_draws) - len(loop_draws) + 1):

            i = 0

            while i < loop_size:

                if rest_draws[diff + i] != loop_draws[i]:
                    break
                i += 1

            if i == loop_size:
                return True

        loop_size += 1

    return False


# start training
main()