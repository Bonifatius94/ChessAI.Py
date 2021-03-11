
# TODO: do a complete rework with a Monte Carlo approach, no deep learning etc.

import os
import tensorflow as tf
import chesslib
import random
import numpy as np
import math


# configure tensorflow logging
#import logging
#logging.getLogger('tensorflow').setLevel(logging.FATAL)


# define hyper-params
params = {
    'training_epoch': 100,
    'learning_rate': 0.01,
    'results_out_dir': os.environ['MODEL_OUT'],
    'train_steps_per_epoch': 1000,
    'eval_steps_per_epoch': 50,
}


def create_model():

    # create new keras model instance with the given fully-connected neuronal layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # TODO: try LSTM cells instead of fully-connected layers to learn best draws for the given context

    # prepare the model for learning with chess board batches
    model.build(input_shape=(None, 14))
    return model


# init keras model for first iteration
model_train = create_model()
model_cache = create_model()

# define optimizer and loss function for training
optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
loss_func = tf.losses.MeanSquaredError()


def main():

    print("starting chess reinforcement learning")

    # init training
    epoch = 1
    prepare_models()

    # do endless training loop
    while True:

        print("starting epoch", epoch)

        for step in range(params['train_steps_per_epoch']):
            train_step()

        

        epoch += 1


def prepare_models():

    # load 'best' model weights if weights are saved
    pretrained_weights_file = os.path.join(params['results_out_dir'], 'pre_trained_weights.h5')
    if os.path.exists(pretrained_weights_file):
        model_train.load_weights(pretrained_weights_file)
        model_cache.load_weights(pretrained_weights_file)
    elif os.path.isdir(results_out_dir):
        files = os.listdir(results_out_dir)
        epoch = -1
        for file in files:
            if (file.startswith('chess-ai-model-weights')):
                end_index = file.find('.')
                epoch = max([epoch, int(file[23:end_index])])
        if epoch > -1:
            print("loading existing model from epoch ", epoch)
            load_model_weights(model_train, epoch)
            load_model_weights(model_cache, epoch)


@tf.function
def train_step(batch_data):

    with tf.GradientTape() as tape:

        # get logits output
        features, labels = batch_data
        logits = model_train(features, True)
        loss = loss_func(logits, labels)

    # compute gradients and optimize the model
    gradients = tape.gradient(loss, model_train.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_train.trainable_variables))
    return logits


def test_step():

    # TODO: implement test logic (play vs. stockfish engine)
    return 0.0


def save_model(model: tf.keras.Model):

    # serialize model to JSON
    model_json = model.to_json()
    file_path = os.path.join(results_out_dir, "chess-ai-model.json")
    with open(file_path, "w") as json_file:
        json_file.write(model_json)


def save_model_weights(model: tf.keras.Model, epoch: int):

    # serialize weights to HDF5
    file_name = "chess-ai-model-weights-{}.h5".format(epoch)
    file_path = os.path.join(results_out_dir, file_name)
    model.save_weights(file_path)


def load_model_weights(model: tf.keras.Model, epoch: int):

    # serialize weights to HDF5
    file_name = "chess-ai-model-weights-{}.h5".format(epoch)
    file_path = os.path.join(results_out_dir, file_name)
    model.load_weights(file_path)


def do_training(best_estimator: tf.keras.Model, mutated_estimator: tf.keras.Model):

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


def get_reward(game_state: str):

    if game_state == 'w': # win
        return 1
    elif game_state == 't': # tie
        return 0
    elif game_state == 'd': # loss
        return -1


def play_chessgame(best_estimator: tf.keras.Model, mutated_estimator: tf.keras.Model, training_side):

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
        fill_column = np.expand_dims(draws, axis=1)
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
if __name__ == '__main__':
    main()