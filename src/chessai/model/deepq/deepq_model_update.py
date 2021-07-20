
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from chessai.core import AbstractModelAdjuster


class ChessDeepQModelAdjuster(AbstractModelAdjuster):

    def __init__(self, alpha: float=0.01, gamma: float=0.99):
        super(ChessDeepQModelAdjuster, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.optimizer = Adam(learning_rate=self.alpha)


    def update_weights(self, model: tf.keras.Model, train_batch):

        # unpack batch items
        states, actions, rewards, next_states, terminals = train_batch

        # compute labels (= best possible state value after chosen action)
        target_preds = np.max(model(next_states).numpy(), axis=1)
        q_labels = rewards + (1 - terminals) * (target_preds * self.gamma)
        q_labels = q_labels.astype(np.float32)

        # collect all gradients during loss computation
        with tf.GradientTape() as tape:

            # predict the Q values for the given states
            q_preds = model(states, training=True)

            # compute the loss using the MSE between labels and predictions
            loss = tf.reduce_mean(tf.pow(q_labels - q_preds, 2), axis=1)

        # fit the model weights using gradient descent
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))