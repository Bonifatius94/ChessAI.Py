
import tensorflow as tf
from abc import ABC, abstractmethod


class AbstractModelAdjuster(ABC):
    """
    An abstract class providing a model update functionality
    used for training a kinds of neural networks, etc.
    """

    @abstractmethod
    def update_weights(self, model: tf.keras.Model, train_batch: tuple):
        """
        Update the given model by training on the given batch.

        :param model: The model to be updated.
        :type model: tf.keras.Model
        :param train_batch: The batch to be trained on.
        :type train_batch: tuple
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError('The child class did not implement this function!')
