
import sqlite3
import requests

import numpy as np
import tensorflow as tf
from tensorflow import keras
import chesslib


training_input = []
training_labels = []
test_input = []
test_labels = []


def main():

    # load training data and train the model
    load_training_data()
    do_training()


def load_training_data():

    global training_input
    global training_labels
    global test_input
    global test_labels

    # download the training data set
    file = 'win_rates.db'
    url = 'https://raw.githubusercontent.com/Bonifatius94/ChessAI.CS/master/Chess.AI/Data/win_rates.db'
    req = requests.get(url, allow_redirects=True)
    open(file, 'wb').write(req.content)

    # query training data
    conn = sqlite3.connect(file)
    cursor = conn.cursor()
    cursor.execute('SELECT BoardBeforeHash, DrawHashNumeric, WinRate FROM WinRateInfo WHERE AnalyzedGames >= 10')
    win_rates_cache = cursor.fetchall()
    conn.close()

    # shuffle win rates for training randomization
    win_rates_cache = np.array(win_rates_cache)
    np.random.shuffle(win_rates_cache)

    # init label dataset
    label_dataset = np.array([x[2] for x in win_rates_cache], dtype=np.float)
    print(label_dataset)

    # init input dataset
    boards = np.array([
        chesslib.ApplyDraw(
            chesslib.Board_FromHash(np.frombuffer(bytes.fromhex(x[0]), dtype=np.uint8)),
            int(x[1]))
        for x in win_rates_cache],
        dtype=np.uint64)
    draws = np.array(win_rates_cache[:, 1], np.uint64)
    draws = np.expand_dims(draws, axis=1)
    input_dataset = np.append(boards, draws, axis=1)

    # assign datasets and labels
    separator = int(input_dataset.shape[0] * 0.9)
    training_input = input_dataset[:separator]
    training_labels = label_dataset[:separator]
    test_input = input_dataset[separator:]
    test_labels = label_dataset[separator:]


def do_training():

    # init network to be trained
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(14,)),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'),
        keras.layers.Dense(1, activation='linear')
    ])

    # compile the network
    model.compile(
        optimizer=keras.optimizers.SGD(10e-4),
        loss='mean_squared_error'
    )

    min_loss = float('inf')
    timeout = 10

    # do training until there is no further progress
    while timeout:

        # do the training
        model.fit(training_input, training_labels, epochs=10, batch_size=100)

        # evaluate the training quality
        eval_loss = round(model.evaluate(test_input, test_labels, verbose=2), 4)

        # export the new model weights if there was made an improvement
        if eval_loss < min_loss:

            model.save_weights('/home/ai/model/pre_trained_weights.h5')
            min_loss = eval_loss
            timeout = 10

        timeout -= 1


# start training
main()
