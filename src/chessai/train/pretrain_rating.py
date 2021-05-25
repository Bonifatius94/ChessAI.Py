
from datetime import datetime
import numpy as np
import chesslib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy

from chessai.dataset import ChessGmGamesDataset
from chessai.model.pretrain import ChessRatingModel


class RatingTrainingSession(object):

    def __init__(self, params: dict):

        self.params = params

        # initialize model and datasets
        self.train_dataset, self.eval_dataset = self.load_datasets(params)
        self.model = self.create_model(params)
        print(self.model.summary())

        # create model checkpoint manager
        self.model_ckpt_callback = ModelCheckpoint(
            filepath='/app/model/pretrain-ratings', save_weights_only=True,
            monitor='val_accuracy', mode='max', save_best_only=True)

        # create tensorboard logger
        logdir = '/app/logs/pretrain_ratings/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = TensorBoard(log_dir=logdir)


    def load_datasets(self, params: dict) -> (tf.data.Dataset, tf.data.Dataset):

        # load batched SARS datasets
        dataset_generator = ChessGmGamesDataset(params['batch_size'])
        train_data, eval_data = dataset_generator.load_datasets(0)

        # extract inputs as the chess board (after) and targets as the reward
        # drop the rest of the SARS data, it's not required for this training
        train_data = train_data.map(lambda s0, a, r, s1: (s1, r))
        eval_data = eval_data.map(lambda s0, a, r, s1: (s1, r))

        # redefine the labels as 100 classes uniformly over 0.0 - 1.0 score
        num_classes = params['rating_classes']
        train_data = train_data.map(lambda s1, r: (s1, tf.one_hot(
            tf.cast(r * num_classes, dtype=tf.int32), depth=num_classes)))
        eval_data = eval_data.map(lambda s1, r: (s1, tf.one_hot(
            tf.cast(r * num_classes, dtype=tf.int32), depth=num_classes)))

        # print first training batch to console
        test_batch = next(iter(train_data))
        print('test batch: {}, {}'.format(test_batch[0].numpy(), test_batch[1].numpy()))

        return train_data, eval_data


    def create_model(self, params: dict) -> tf.keras.Model:

        # create learning rate decay func
        lr_decay_func = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params['learn_rate'],
            decay_steps = params['total_train_batches'] * params['lr_decay_epochs'],
            decay_rate = params['lr_decay_rate'],
            staircase=params['lr_decay_staircase']
        )

        # create optimizer and loss func
        optimizer = tf.optimizers.SGD(learning_rate=lr_decay_func, momentum=0.9)
        loss_func = tf.losses.CategoricalCrossentropy()

         # create model to be trained
        model = ChessRatingModel(params)
        model.build((None, 8, 8, 7))
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

        return model


    def run_training(self) -> tf.keras.callbacks.History:

        # run the training and write the metrics to tensorboard
        history = self.model.fit(
            x=self.train_dataset, epochs=self.params['epochs'],
            validation_data=self.eval_dataset,
            callbacks=[self.tb_callback, self.model_ckpt_callback])

        return history
