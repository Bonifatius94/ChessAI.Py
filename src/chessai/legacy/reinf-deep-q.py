
import tensorflow as tf
import numpy as np


class ChessModel(tf.keras.Model):

    def __init__(self, params: dict):

        super(ChessModel, self).__init__()
        self.params = params

        # TODO: define layers
        pass


    def call(self, inputs):

        # TODO: define dataflow between layers
        pass





















# Common Deep Q Learning:
# =======================
# Train a model to estimate Q values for each (state, action) tuple. 
# The Q values determine the goodness of an action for a given state (by estimating the action's reward).
# Those Q values can lateron be used as draw rating heuristics.

# Applying Deep Q Learning to Chess:
# ==================================
# In chess, a state equals the chess board at a given time and an actions is a legal draw.
# The training can be done from scratch with self-play.

# Neuronal Network:
# =================
# input: chess board as 64 bytes array, chess draw as 16 bit short integer (2 bytes) -> 66 bytes input (normalized values within [-1; 1])
# some dense layers, maybe also some Conv / LSTM layers
# output: est. Q values of each draw within [0.0; 1.0]

# DDQN Algorithm:
# ===============
# 2 deep Q networks: one for training, one as target (stabilizing cache)
# Choose actions according to the target network's Q table
# Include model learning: make the training network learn to predict the Q values more accurately by repacturing experience from a cache memory
# 

# Play Game:
# ==========
# choose sides
# until the game is over (do-while):
#   compute all possible draws
#   evaluate the scores of the draws with the est. func. of the drawing side's player
#   pick the 'best' draw and apply it to the board

# Rewarding / Punishment:
# =======================
# game won  -> 1.0
# game tied -> 0.5
# game lost -> 0.0

# Evaluation Phase:
# =================
# play against well-known strong chess engines like Stockfish
# -> determine if the learned behavior is actually successful