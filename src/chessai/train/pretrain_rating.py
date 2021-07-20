
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from chessai.dataset import ChessGmGamesDataset
from chessai.model.pretrain import ChessRatingModel


class RatingTrainingSession(object):

    def __init__(self, params: dict):

        print('creating rating pre-training session ...')

        self.params = params

        # initialize model and datasets
        print('loading GM games dataset ...')
        self.train_dataset, self.eval_dataset = self.load_datasets(params)
        print('done!')

        print('creating model ...')
        self.model = self.create_model(params)
        # print(self.model.summary())

        # create model checkpoint manager
        self.model_ckpt_callback = ModelCheckpoint(
            filepath='/app/model/pretrain-ratings/', save_weights_only=True,
            monitor='val_loss', mode='max', save_best_only=True)

        # create tensorboard logger
        logdir = '/app/logs/pretrain_ratings/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)

        print('done!')


    def load_datasets(self, params: dict):

        # load batched SARS datasets
        dataset_generator = ChessGmGamesDataset(params['batch_size'])
        train_data, eval_data = dataset_generator.load_datasets(
            min_occurrences=params['min_occ'])

        # extract inputs as the chess board (after) and targets as the reward
        # drop the rest of the SARS data, it's not required for this training
        train_data = train_data.map(lambda s0, a, r, s1, side: ((s1, side), r))
        eval_data = eval_data.map(lambda s0, a, r, s1, side: ((s1, side), r))

        # print first training batch to console
        # test_batch = next(iter(train_data))
        # print('test batch: {}, {}'.format(test_batch[0].numpy(), test_batch[1].numpy()))

        return train_data, eval_data


    def create_model(self, params: dict) -> tf.keras.Model:

        # # create learning rate decay func
        # lr_decay_func = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=params['learn_rate'],
        #     decay_steps = params['total_train_batches'] * params['lr_decay_epochs'],
        #     decay_rate = params['lr_decay_rate'],
        #     staircase=params['lr_decay_staircase']
        # )

        # create optimizer and loss func
        # optimizer = tf.optimizers.SGD(learning_rate=lr_decay_func, momentum=0.9)
        optimizer = tf.optimizers.Adam(lr=params['learn_rate'])
        loss_func = tf.losses.MSE

         # create model to be trained
        model = ChessRatingModel(params)
        #model.build((None, 8, 8, 13))
        model.compile(optimizer=optimizer, loss=loss_func)

        return model


    def run_training(self) -> tf.keras.callbacks.History:

        print('starting rating pre-training ...')

        # run the training and write the metrics to tensorboard
        history = self.model.fit(
            x=self.train_dataset, epochs=self.params['epochs'],
            validation_data=self.eval_dataset,
            callbacks=[self.tb_callback, self.model_ckpt_callback])

        print('training done!')

        return history
