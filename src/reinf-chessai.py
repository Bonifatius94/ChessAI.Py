
import tensorflow as tf
from tensorflow import keras
import chesslib


def main():

    # start the training
    do_training()


def do_training():

    # TODO: implement training function

    # neuronal network:
    # =================
    # input: 14 x 64-bit integer array
    # 7 layers with 128 neurons, sigmoid act. func.
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


# start main function
main()