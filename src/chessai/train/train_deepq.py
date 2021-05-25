
from datetime import datetime
import numpy as np
import chesslib, sys

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from chessai.dataset import ChessESoftTreewalkDataset
from chessai.model.deepq import ChessDeepQModel
from chessai.benchmark import StockfishBenchmark
from chessai.dataset import convert_states


class DeepQTrainingSession(object):

    def __init__(self, params: dict):

        self.params = params

        # initialize the model
        self.model = self.create_model(params)
        print(self.model.summary())

        # initialize the benchmark metric (stockfish)
        self.benchmark = StockfishBenchmark(params['stockfish_level'])

        # create model checkpoint manager
        self.model_ckpt_callback = ModelCheckpoint(
            filepath='/app/model/deep-q', save_weights_only=True,
            monitor='val_accuracy', mode='max')

        # create tensorboard logger
        logdir = '/app/logs/deep-q/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def generate_dataset(self, value_est: tf.keras.Model, params: dict) -> tf.data.Dataset:

        # load a batched self-play SARS dataset
        dataset_generator = ChessESoftTreewalkDataset()
        dataset = dataset_generator.load_dataset(
            value_est=value_est,
            epsilon=params['expl_rate'],
            batch_size=params['batch_size']
        )

        # estimate the values for each state
        dataset = dataset.map(lambda s0, a, r, s1, t: (
            s0,
            r - params['gamma'] * (self.model(s1) * -1.0 * (1 - t)) # Q update formula
            # info: * -1 is required not to learn actions beneficial for the opponent
        ))

        return dataset


    def create_model(self, params: dict) -> tf.keras.Model:

        # create optimizer and loss func
        momentum = params['momentum'] if 'momentum' in params else 0.0
        optimizer = SGD(learning_rate=params['learn_rate'], momentum=momentum)
        loss_func = tf.losses.MSE

         # create model to be trained
        model = ChessDeepQModel(params)
        model.build((None, 8, 8, 7))
        model.compile(optimizer=optimizer, loss=loss_func)

        return model


    def run_training(self):

        # loop through all training epochs
        for epoch in range(self.params['epochs']):

            # generate a training dataset by self-play
            selfplay_dataset = self.generate_dataset(self.model, self.params)

            # clone the model (training model needs to be different to 
            # the model used for decision making in the dataset generator)
            # weights = self.model.get_weights()
            # model_train = self.create_model(self.params)
            # model_train.set_weights(weights)

            # learn the TD errors of the generated dataset
            self.model.fit(
                x=selfplay_dataset, epochs=self.params['fit_epochs'],
                steps_per_epoch=self.params['batches_per_epoch'],
                callbacks=[self.tb_callback, self.model_ckpt_callback])
 
            # override the old model for prediction with the new model
            # self.model = model_train

            # # test the model after several training epochs
            # if (epoch + 1) % self.params['sample_interval'] == 0:
            #     self.greedy_sampling()

            # TODO: enable this when the stockfish benchmark is ready for use
            # if (epoch + 1) % 100 == 0:
            #     # benchmark the training progress
            #     choose_action = lambda s0, a0: self.choose_action(s0, a0)
            #     win_rate, tie_rate, loss_rate = self.benchmark.evaluate(
            #         params['benchmark_games'], choose_action)
            #     tf.print('benchmark results (w/t/l): {} / {} / {}'.format(
            #         win_rate, tie_rate, loss_rate))


    def greedy_sampling(self):

        # TODO: fix greedy sampling

        # sample a single game with a greedy policy (epsilon=0)
        dataset_generator = ChessESoftTreewalkDataset()
        dataset = dataset_generator.load_dataset(value_est=self.model, 
            epsilon=0.0, batch_size=32, shuffle=False)
        dataset = dataset.unbatch()

        # print the start formation to console
        s0 = chesslib.ChessBoard_StartFormation()
        print(chesslib.VisualizeBoard(s0) + '\n')
        side = 0

        # loop through all timesteps of the game
        for sars_timestep in dataset:

            # unwrap the SARS data of the timestep
            s0, a, r, s1, t = sars_timestep

            # print the selected draw and the board after applying it
            print(chesslib.VisualizeDraw(a))
            print(chesslib.VisualizeBoard(s1) + '\n')

            # alternate side between 0 and 1 (white=0, black=1)
            side = 1 - side

        # print the game's outcome
        final_state = chesslib.GameState(s1, a)
        win_text = ('black' if side else 'white') + ' player wins'
        tie_text = 'game ends with a tie'
        print(win_text if final_state == chesslib.GameState_Checkmate else tie_text)


    def choose_action(self, obs0: np.ndarray, last_action: int):

        # determine the drawing side
        first_draw = last_action == chesslib.ChessDraw_Null
        drawing_side = chesslib.ChessColor_White if first_draw \
            else 1 - (last_action & 0x800000 >> 23)

        # compute all possible actions and apply them to the board
        poss_actions = chesslib.GenerateDraws(obs0, drawing_side, last_action, True)
        poss_obs1 = [chesslib.ApplyDraw(action) for action in poss_actions]
        conv_obs1 = convert_states(np.reshape(poss_obs1, (num_actions, 13)))

        # rate each action using the model and choose the best one
        action_ratings = self.model(conv_obs1).numpy() * -1
        action_id = np.argmax(action_ratings)
        best_action = poss_actions[action_id]

        return best_action
