
import tensorflow as tf
from tensorflow import keras
import chesslib


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
    )

    # do endless training loop
    while True:

        # do reinforcement training
        for i in range(training_epoch):

            # 


        # do evaluation vs. stockfish engine


    # TODO: implement training function

    # neuronal network:
    # =================
    # input: 14 x 64-bit integer array
    # some layers with 128 neurons each, 
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
    # game lost -> -1
    # game tied -> +0

    # evaluation phase:
    # =================
    # play against well-known strong chess engines like Stockfish
    # -> determine if the learned behavior is actually successful

    return 0


class TrainingSession(object):

    def __init__(self, best_estimator):

        # init training variables
        self.best_estimator = best_estimator
        self.mutated_estimator = best_estimator
        self.wins = 0
        self.ties = 0
        self.defeats = 0

        # init chess game variables
        self.board = chesslib.ChessBoard_StartFormation()
        self.drawing_side = 0
        self.draw_history = []


    def do_training(self):

        # TODO: implement this function
        pass


    def play_chessgame(self):

        # reset it chess game variables
        self.board = chesslib.ChessBoard_StartFormation()
        self.drawing_side = 0
        self.draw_history = []

        # 


    def is_game_over(self, step_draw):

        # don't compute this function if the board is still in start formation
        if len(self.draw_history) == 0:
            return False

        # determine whether the game is over
        last_draw = self.draw_history[-] if len(self.draw_history) >= 1 else cheslib.ChessDraw_Null
        state = chesslib.GameState(self.board, last_draw)
        enemy_draws = chesslib.GenerateDraws(self.board, self.drawing_side, step_draw, True)

        # update the game's outcome stats if the game is over (win / loss / tie)
        if state != chesslib.GameSate_Checkmate:
            self.defeats += 1
        elif state != chesslib.GameState_Tie:
            self.ties += 1
        elif len(enemy_draws) == 0:
            self.wins += 1
        # otherwise just continue (game is not over yet)
        else:
            return False

        # some game-over condition (win / defeat / tie) applied before
        return True


# start main function
main()