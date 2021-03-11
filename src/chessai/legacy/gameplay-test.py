
import chesslib
import numpy as np
import math
import sys
import random
import tensorflow
from tensorflow import keras


def do_training(id, best_estimator: keras.Model, mutated_estimator: keras.Model):

    print("start training session #", id)

    wins = 0
    ties = 0
    defeats = 0

    # play several training games
    for i in range(1000):

        # play one game
        game_state = play_chessgame(best_estimator, mutated_estimator)

        # apply the game's outcome
        if game_state == 'w':
            wins += 1
        elif game_state == 'd':
            defeats += 1
        elif game_state == 't':
            ties += 1

        # print_progress(i, 1000)
        print("game", i)

    # determine the win rate
    win_rate = 1.0 * wins / (wins + defeats) if (wins + defeats) > 0 else 0.0

    print("end training session #", id, " win rate:", win_rate)
    return (id, win_rate)


def play_chessgame(best_estimator: keras.Model, mutated_estimator: keras.Model):

    # reset it chess game variables
    board = chesslib.ChessBoard_StartFormation()
    drawing_side = 0
    draw_history = []
    game_state = 'n'

    # determine side selection
    training_side = random.randint(0, 1)
    round = 1.0

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

        # apply the draw to the chessboard
        board = chesslib.ApplyDraw(board, best_draw)

        # update game variables
        draw_history.append(best_draw)

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


def get_game_state(board, draw_history, drawing_side):

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


# init random model
rand_estimator_1 = keras.Sequential([
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

# init random model
rand_estimator_2 = keras.Sequential([
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


# play several games against oneself
do_training(0, rand_estimator_1, rand_estimator_2)