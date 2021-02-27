
import sqlite3
import requests

import numpy as np
import tensorflow as tf
from tensorflow import keras
import chesslib

from gm_games_dataset import load_datasets


class ChessFeatureExtractionModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessFeatureExtractionModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_conv_1 = tf.keras.layers.Conv2D(16, (8, 8), strides=1, padding='same', activation='relu')
        self.nn_conv_2 = tf.keras.layers.Conv2D(16, (6, 6), strides=1, padding='same', activation='relu')
        self.nn_conv_3 = tf.keras.layers.Conv2D(16, (5, 5), strides=1, padding='same', activation='relu')
        self.nn_conv_4 = tf.keras.layers.Conv2D(16, (4, 4), strides=1, padding='same', activation='relu')
        self.nn_conv_5 = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')
        self.nn_conv_6 = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')
        self.nn_flatten_output = tf.keras.layers.Flatten()
        self.nn_dense_output = tf.keras.layers.Dense(units=512)


    def call(self, inputs):

        x = inputs
        x = self.nn_conv_1(x)
        x = self.nn_conv_2(x)
        x = self.nn_conv_3(x)
        x = self.nn_conv_4(x)
        x = self.nn_conv_5(x)
        x = self.nn_conv_6(x)
        x = self.nn_flatten_output(x)
        x = self.nn_dense_output(x)
        return x


# class ChessPretrainDraws(tf.keras.Model):

#     def __init__(self, params: dict):

#         # call super constructor and store overloaded parameters
#         super(ChessPretrainDraws, self).__init__()
#         self.params = params

#         # create model layers
#         self.nn_feature_ext = ChessFeatureExtractionModel()



class ChessRatingModel(tf.keras.Model):

    def __init__(self, params: dict):

        # call super constructor and store overloaded parameters
        super(ChessRatingModel, self).__init__()
        self.params = params

        # create model layers
        self.nn_feature_ext = ChessFeatureExtractionModel(params)
        self.nn_dense_1 = tf.keras.layers.Dense(512)
        self.nn_dense_2 = tf.keras.layers.Dense(256)
        self.nn_dense_3 = tf.keras.layers.Dense(128)
        self.nn_dense_4 = tf.keras.layers.Dense(64)
        self.nn_dense_output = tf.keras.layers.Dense(1)


    def call(self, inputs):

        x = inputs
        x = self.nn_feature_ext(x)
        x = self.nn_dense_1(x)
        x = self.nn_dense_2(x)
        x = self.nn_dense_3(x)
        x = self.nn_dense_4(x)
        x = self.nn_dense_output(x)
        return x


class TrainingSession(object):

    def __init__(self, params: dict):

        super(TrainingSession, self).__init__()
        self.params = params

        # create training and evaluation datasets
        self.train_dataset, self.eval_dataset = load_datasets(params['batch_size'])

        # create model to be trained
        self.model = ChessRatingModel(params)
        self.model.build((None, 8, 8, 7))
        print(self.model.summary())

        # create optimizer and loss func
        self.optimizer = tf.optimizers.SGD(learning_rate=self.params['learn_rate'])
        self.loss_func = tf.losses.MeanSquaredError()

        # create model checkpoints
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, './models/pretrain', max_to_keep=5)

        # create logging metrics
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc = tf.keras.metrics.Mean(name="train_acc")
        self.eval_loss = tf.keras.metrics.Mean(name="eval_loss")
        self.eval_acc = tf.keras.metrics.Mean(name="eval_acc")

        # create TensorBoard summary writer
        self.train_summary_writer = tf.summary.create_file_writer('./logs/train')
        self.eval_summary_writer = tf.summary.create_file_writer('./logs/eval')


    def run_training(self):

        step = 0

        # loop through all epochs
        for epoch in range(self.params['epochs']):

            print('starting training epoch', epoch)

            # loop through all batches on the training dataset
            for batch_data in self.train_dataset:
                self.train_step(batch_data)
                step += 1

                # log training progress and metrics
                if step % self.params['log_interval'] == 0:
                    print('training progress:', int((step % self.params['total_train_batches']) / self.params['total_train_batches'] * 100), '%')
                    with self.train_summary_writer.as_default():
                        print('train results:', 'acc={}, loss={}'.format(self.train_acc.result(), self.train_loss.result()))
                        tf.summary.scalar('train_acc', self.train_acc.result(), step=step)
                        tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                        self.train_acc.reset_states()
                        self.train_loss.reset_states()

            # loop through all batches on the evaluation dataset
            for batch_data in self.eval_dataset:
                self.eval_step(batch_data)

            with self.eval_summary_writer.as_default():
                print('eval results:', 'acc={}, loss={}'.format(self.eval_acc.result(), self.eval_loss.result()))
                tf.summary.scalar('eval_acc', self.eval_acc.result(), step=step)
                tf.summary.scalar('eval_loss', self.eval_loss.result(), step=step)
                self.eval_acc.reset_states()
                self.eval_loss.reset_states()

            # store the model weights using the checkpoint manager
            self.manager.save(epoch)


    @tf.function
    def train_step(self, batch_data):

        # unwrap batch data as SARS data
        states, actions, rewards, next_states = batch_data

        with tf.GradientTape() as tape:

            # TODO: figure out if next states should rather be used for prediction

            # predict ratings and compute loss
            pred_ratings = self.model(next_states)
            label_ratings = rewards
            loss = self.loss_func(y_true=label_ratings, y_pred=pred_ratings)

        # compute gradients and optimize the model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # write loss and accuracy to the metrics cache
        self.train_loss(loss)
        self.train_acc(1 - tf.math.sqrt(loss))


    @ tf.function
    def eval_step(self, batch_data):

        # unwrap batch data as SARS data
        states, actions, rewards, next_states = batch_data

        # predict ratings and compute loss
        pred_ratings = self.model(next_states)
        label_ratings = rewards
        loss = self.loss_func(y_true=label_ratings, y_pred=pred_ratings)

        # write loss and accuracy to the metrics cache
        self.eval_loss(loss)
        self.eval_acc(1 - tf.math.sqrt(loss))


def main():

    params = {
        'batch_size': 32,
        'learn_rate': 0.01,
        'epochs': 1000,

        'log_interval': 100,
        'total_train_batches': 2774,
    }

    session = TrainingSession(params)
    session.run_training()


# launch training
if __name__ == '__main__':
    main()
