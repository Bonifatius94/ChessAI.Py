
import sqlite3
import requests

import numpy as np
import tensorflow as tf
from tensorflow import keras
import chesslib

from chessai.dataset import ChessGmGamesDataset
from chessai.pretrain import ChessDrawGenerator


class DrawGenTrainingSession(object):

    def __init__(self, params: dict):

        super(DrawGenTrainingSession, self).__init__()
        self.params = params

        # create training and evaluation datasets
        dataset = ChessGmGamesDataset(params['batch_size'])
        self.train_dataset, self.eval_dataset = dataset.load_datasets()

        # create model to be trained
        self.model = ChessDrawGenerator(params)
        # self.model.build((None, 8, 8, 7))
        # print(self.model.summary())

        # create learning rate decay func
        self.lr_decay_func = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params['learn_rate'],
            decay_steps = params['total_train_batches'] * params['lr_decay_epochs'],
            decay_rate = params['lr_decay_rate'],
            staircase=False
        )

        # create optimizer and loss func
        self.optimizer = tf.optimizers.SGD(learning_rate=self.lr_decay_func)
        self.loss_func = tf.losses.CategoricalCrossentropy()

        # create model checkpoints
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, '/app/models/pretrain-fx', max_to_keep=5)

        # create logging metrics
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc = tf.keras.metrics.Mean(name="train_acc")
        self.eval_loss = tf.keras.metrics.Mean(name="eval_loss")
        self.eval_acc = tf.keras.metrics.Mean(name="eval_acc")

        # create TensorBoard summary writer
        self.train_summary_writer = tf.summary.create_file_writer('/app/logs/train')
        self.eval_summary_writer = tf.summary.create_file_writer('/app/logs/eval')


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
                        # tf.summary.scalar('train_acc', self.train_acc.result(), step=step)
                        tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                        # self.train_acc.reset_states()
                        self.train_loss.reset_states()

            # loop through all batches on the evaluation dataset
            for batch_data in self.eval_dataset:
                self.eval_step(batch_data)

            with self.eval_summary_writer.as_default():
                print('eval results:', 'acc={}, loss={}'.format(self.eval_acc.result(), self.eval_loss.result()))
                # tf.summary.scalar('eval_acc', self.eval_acc.result(), step=step)
                tf.summary.scalar('eval_loss', self.eval_loss.result(), step=step)
                # self.eval_acc.reset_states()
                self.eval_loss.reset_states()

            # store the model weights using the checkpoint manager
            self.manager.save(epoch)


    @tf.function
    def train_step(self, batch_data):

        # unwrap batch data as SARS data
        states, actions, rewards, next_states = batch_data
        input_positions = tf.bitwise.bitwise_and(actions, 0x3F)
        target_positions = tf.bitwise.bitwise_and(actions, 0xFC0)
        target_positions = tf.one_hot(target_positions, depth=64, dtype=tf.float32)

        with tf.GradientTape() as tape:

            # initialize the LSTM states by showing the chess boards to it
            h = self.model.show_chessboard(states)

            # now, predict the most likely target position for the acting chess pieces
            pred_positions = tf.nn.softmax(self.model(input_positions, h))

            # compute the loss (check if the correct target positions were predicted)
            loss = self.loss_func(y_true=target_positions, y_pred=pred_positions)

            # scale the loss according to the rewards (-> favour better actions)
            # loss = loss * tf.math.log(1 - tf.cast(rewards, dtype=tf.float32))

        # compute gradients and optimize the model
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # write loss and accuracy to the metrics cache
        self.train_loss(loss)
        # self.train_acc(1 - loss)


    @ tf.function
    def eval_step(self, batch_data):

        # unwrap batch data as SARS data
        states, actions, rewards, next_states = batch_data
        input_positions = tf.bitwise.bitwise_and(actions, 0x3F)
        target_positions = tf.bitwise.bitwise_and(actions, 0xFC0)
        target_positions = tf.one_hot(target_positions, depth=64, dtype=tf.float32)

        # initialize the LSTM states by showing the chess boards to it
        h = self.model.show_chessboard(states)

        # now, predict the most likely target position for the acting chess pieces
        pred_positions = tf.nn.softmax(self.model(input_positions, h))

        # compute the loss (check if the correct target positions were predicted)
        # scale the loss according to the rewards (-> favour better actions)
        loss = self.loss_func(y_true=target_positions, y_pred=pred_positions)
        # loss = loss * tf.math.log(1 - rewards)

        # write loss and accuracy to the metrics cache
        self.eval_loss(loss)
        # self.eval_acc(1 - loss)
