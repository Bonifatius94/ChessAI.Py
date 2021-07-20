
from datetime import datetime
import numpy as np
import chesslib, sys

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from chessai.benchmark import StockfishBenchmark
from chessai.core.chess_game_env import ChessGameEnv
from chessai.dataset import convert_states
from chessai.model.deepq.deepq_agent import ChessDeepQAgent
from chessai.model.deepq.deepq_model_update import ChessDeepQModelAdjuster
from chessai.model.pretrain import drawgen_model


class DeepQTrainingSession(object):

    def __init__(self, params: dict):

        self.params = params

        # # initialize the model
        # self.model = self.create_model(params)
        # print(self.model.summary())

        # initialize the benchmark metric (stockfish)
        self.benchmark = StockfishBenchmark(params['stockfish_level'])

        # # create model checkpoint manager
        # self.model_ckpt_callback = ModelCheckpoint(
        #     filepath='/app/model/deep-q', save_weights_only=True,
        #     monitor='val_accuracy', mode='max')

        # # create tensorboard logger
        # logdir = '/app/logs/deep-q/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.tb_callback = TensorBoard(log_dir=logdir)


    def run_training(self):

        # create the game environment
        env = ChessGameEnv(max_steps_per_episode=self.params['max_steps_per_episode'])

        # create the learning agent
        model_adj = ChessDeepQModelAdjuster(alpha=self.params['learn_rate'], gamma=self.params['gamma'])
        agent = ChessDeepQAgent(env, model_adj, epsilon=self.params['expl_rate'])

        # TODO: think of training 2 different agents representing white / black side

        # create an empty draw history list (used only for metrics)
        draw_history = list()

        # loop through all training epochs
        while env.episode < self.params['epochs']:

            # choose an action and apply it to the game environment
            obs0 = env.state
            action = agent.choose_action()
            obs1, reward, is_terminal = env.step(action)

            # sample a training batch and train on it
            exp = (obs0, action, reward, obs1, is_terminal)
            agent.train_step(exp)

            # log the learning progress (after each episode)
            draw_history.append(action)
            if is_terminal:
                winning_side = 'white' if len(draw_history) % 2 == 0 else 'black'
                print(f'episode: {env.episode}, reward: {reward}')
                print(f'draw history: {draw_history}, winning side: {winning_side}')
                draw_history = list()

            # TODO: add save model calls to persist the trained weights
            # TODO: add evaluation metrics like stockfish benchmark, etc.
