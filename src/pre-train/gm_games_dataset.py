
import sqlite3
import requests
import os, sys

import numpy as np
import pandas as pd
import tensorflow as tf
import chesslib


def load_datasets(batch_size: int):

    # make sure the winrates database is locally available (download if not)
    db_filepath = './win_rates.db'
    download_winrates_db(db_filepath)

    # load the training data into a pandas dataframe using a SQL query
    # then, convert the dataframe to SARS format
    winrates_cache = load_pandas_winrates(db_filepath)
    sars_winrates = prepare_pandas_data(winrates_cache)

    # split the data into randomly sampled training and evaluation chunks (9:1)
    train_data = sars_winrates.sample(frac=0.9)
    eval_data = sars_winrates.drop(train_data.index)

    # transform the pandas dataframe into a tensorflow dataset
    train_dataset = create_tf_dataset(train_data, batch_size, train=True)
    eval_dataset = create_tf_dataset(eval_data, batch_size, train=False)

    return train_dataset, eval_dataset


def download_winrates_db(db_filepath: str):

    # only download the db file if it does not already exist
    if not os.path.exists(db_filepath) or not os.path.isfile(db_filepath):

        # download the sqlite database file and write it to the given local filepath
        url = 'https://raw.githubusercontent.com/Bonifatius94/ChessAI.CS/master/Chess.AI/Data/win_rates.db'
        req = requests.get(url, allow_redirects=True)
        with open(db_filepath, 'wb') as out_file:
            out_file.write(req.content)


def load_pandas_winrates(db_filepath: str):

    conn = sqlite3.connect(db_filepath)
    sql_str = 'SELECT BoardBeforeHash, DrawHashNumeric, WinRate \
        FROM WinRateInfo -- WHERE AnalyzedGames >= 10'
    win_rates_cache = pd.read_sql(sql_str, conn)
    conn.close()
    return win_rates_cache


def prepare_pandas_data(pd_winrates: pd.DataFrame):

    # convert the dataframe into SARS data slices (state-action-reward-nextstate)
    states = [bitboards_from_hash(x) for x in pd_winrates['BoardBeforeHash']]
    actions = [compact_draw_from_hash(x) for x in pd_winrates['DrawHashNumeric']]
    #actions = [int(x) for x in pd_winrates['DrawHashNumeric']]
    #compact_actions = [compact_draw_from_hash(x) for x in pd_winrates['DrawHashNumeric']]
    rewards = pd_winrates['WinRate']

    # combine the SARS data slices to a dataframe
    pd_winrates_sars = pd.DataFrame()
    pd_winrates_sars['states'] = states
    pd_winrates_sars['actions'] = actions
    #pd_winrates_sars['compact_actions'] = compact_actions
    pd_winrates_sars['rewards'] = rewards

    next_states = pd_winrates_sars.apply(lambda x: chesslib.ApplyDraw(x[0], x[1]), axis=1)
    pd_winrates_sars['next_states'] = next_states

    print(pd_winrates_sars.head())

    return pd_winrates_sars


def bitboards_from_hash(board_hash: str):
    return chesslib.Board_FromHash(np.frombuffer(bytes.fromhex(board_hash), dtype=np.uint8))


def compact_draw_from_hash(draw_hash: str):
    return int(draw_hash) & 0x7FFF


def create_tf_dataset(pd_sars_winrates: pd.DataFrame, batch_size: int, train: bool):

    # convert the pandas dataframe into a tensor dataset
    dataset = tf.data.Dataset.from_tensor_slices(pd_sars_winrates)

    # map the chesslib bitboards to convolutable 8x8x7 feature maps
    dataset = dataset.map(lambda x: (
        chessboard_to_feature_maps(x[0]),    # state
        x[1],                                # action
        x[2],                                # reward
        chessboard_to_feature_maps(x[3])))   # next state

    # batch the data properly
    dataset = dataset.batch(batch_size)

    # shuffle the dataset when creating a training dataset
    if train: dataset = dataset.shuffle(5000)

    return dataset


@tf.function
def chessboard_to_feature_maps(bitboards):

    # create empty tensor array for bitboards
    feature_maps = tf.TensorArray(dtype=tf.float32, size=13, dynamic_size=False)

    # loop through all bitboards
    for i in tf.range(tf.size(bitboards)):

        # load single bitboard and create a feature map to write to
        temp_bitboard = bitboards[i]
        temp_feature_map = tf.TensorArray(dtype=tf.float32, size=64, dynamic_size=False)

        # loop through all bitboard positions
        for pos in tf.range(64):
            # extract the piece_set bit from the bitboard: (bitboard >> pos) & 1
            bit = tf.bitwise.bitwise_and(tf.bitwise.right_shift(temp_bitboard, pos), 1)
            temp_feature_map.write(pos, tf.cast(bit, tf.float32))

        # reshape the 64 bitboard positions to 8x8 format, dimensions are (row, column)
        temp_feature_map = tf.reshape(temp_feature_map, (8, 8))

        # apply the converted feature map to the output
        feature_maps.write(i, temp_feature_map)

    return feature_maps


@tf.function
def chessboard_to_2d_feature_maps(bitboards):

    # extract each bit from the bitboards -> bring it into 13x8x8 shape
    feature_maps = chessboard_to_feature_maps(bitboards)

    # union the pieces of each type onto a single feature map and encode it
    # as follows: white piece = 1, black piece = -1, nothing = 0
    # the resulting maps are of shape 6x8x8
    unified_piece_maps = feature_maps[0:6, :, :] + (feature_maps[6:12, :, :] * -1.0)

    # append the was_moved bitboard to piece maps -> 6x8x8 shape
    compressed_feature_maps = tf.concat((unified_piece_maps, feature_maps[12:13, :, :]), axis=0)

    # transpose the compressed feature maps, so the content can be convoluted
    # by field positions and maps are the channels -> final shape 8x8x7
    return tf.transpose(compressed_feature_maps, (1, 2, 0))


if __name__ == '__main__':

    # test loading training and eval datasets
    batch_size = 32
    train_dataset, eval_dataset = load_datasets(batch_size)

    print(train_dataset)
    print(eval_dataset)