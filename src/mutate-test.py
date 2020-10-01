
import numpy as np
import math
import sys
import random
import tensorflow as tf
from tensorflow import keras


def mutate_model(orig_model: keras.Model, learning_rate: float):

    # clone the given model
    model = tf.keras.models.clone_model(orig_model)

    # get the model's weights
    weights = model.get_weights()

    for i in range(len(weights)):

        # calculate the differential weight changes (random uniform distribution)
        shape = weights[i].shape
        updates = np.random.uniform(-learning_rate, learning_rate, shape)

        old_weights = weights[i]
        updated_weights = np.add(weights[i], updates)

        upper_bound = updated_weights >= -1
        lower_bound = updated_weights <= 1
        within_range = np.logical_and(upper_bound, lower_bound)
        outside_range = np.logical_not(within_range)

        if len(old_weights[outside_range]) > 0:
            updated_weights = np.add(updated_weights[within_range], old_weights[outside_range])

        # apply updates to the weights
        weights[i] = updated_weights

    # update the model with the new weights and return the updated model
    model.set_weights(weights)

    return model


# init random model
rand_estimator = keras.Sequential([
    keras.layers.Flatten(input_shape=(14,)),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
    keras.layers.Dense(1, activation='linear')
])

# mutate the random model
mutated_estimator = mutate_model(rand_estimator, 0.01)

print("original", rand_estimator.get_weights())
print("mutation", mutated_estimator.get_weights())