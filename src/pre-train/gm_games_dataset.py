
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
    actions = pd_winrates['DrawHashNumeric']
    rewards = pd_winrates['WinRate']

    # combine the SARS data slices to a dataframe
    pd_winrates_sars = pd.DataFrame()
    pd_winrates_sars['states'] = states
    pd_winrates_sars['actions'] = actions
    pd_winrates_sars['rewards'] = rewards

    next_states = pd_winrates_sars.apply(lambda x: chesslib.ApplyDraw(x[0], x[1]), axis=1)
    pd_winrates_sars['next_states'] = next_states

    return pd_winrates_sars


def bitboards_from_hash(board_hash: str):
    return chesslib.Board_FromHash(np.frombuffer(bytes.fromhex(board_hash), dtype=np.uint8))


def compact_draw_from_hash(draw_hash: str):
    return int(draw_hash) & 0x7FFF


def create_tf_dataset(pd_sars_winrates: pd.DataFrame, batch_size: int, train: bool):

    # convert the pandas dataframe's columns to tensor slices
    states = tf.convert_to_tensor(np.array(
        [np.squeeze(np.array(x)) for x in pd_sars_winrates['states']]), dtype=tf.uint64)
    actions = tf.convert_to_tensor(np.array(pd_sars_winrates['actions']))
    rewards = tf.convert_to_tensor(np.array(pd_sars_winrates['rewards']))
    next_states = tf.convert_to_tensor(np.array(
        [np.squeeze(np.array(x)) for x in pd_sars_winrates['next_states']]), dtype=tf.uint64)

    # create a dataset from the tensor slices -> SARS tuples
    dataset = tf.data.Dataset.from_tensor_slices(
        tensors=(states, actions, rewards, next_states))

    # convert the chesslib bitboards to compressed, convolutable 8x8x7 feature maps
    dataset = dataset.map(lambda state, action, reward, next_state: (
        chessboard_to_2d_feature_maps(state),
        action,
        reward,
        chessboard_to_2d_feature_maps(next_state)
    ))

    # batch the data properly
    dataset = dataset.batch(batch_size)

    # shuffle the dataset when creating a training dataset
    if train: dataset = dataset.shuffle(50)

    return dataset


@tf.function
def chessboard_to_2d_feature_maps(bitboards):

    # extract each bit from the bitboards -> bring it into 13x8x8 shape
    feature_maps = _chessboard_to_feature_maps(bitboards)

    # union the pieces of each type onto a single feature map and encode it
    # as follows: white piece = 1, black piece = -1, nothing = 0
    # the resulting maps are of shape 6x8x8
    unified_piece_maps = feature_maps[0:6, :, :] + (feature_maps[6:12, :, :] * -1.0)

    # append the was_moved bitboard to piece maps -> 6x8x8 shape
    compressed_feature_maps = tf.concat((unified_piece_maps, feature_maps[12:13, :, :]), axis=0)

    # transpose the feature maps, so the content can be convoluted
    # by field positions and maps are the channels -> shape 8x8x7
    transp_feature_maps = tf.transpose(compressed_feature_maps, (1, 2, 0))
    return transp_feature_maps


@tf.function
def _chessboard_to_feature_maps(bitboards):

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
            bit = tf.bitwise.bitwise_and(tf.bitwise.right_shift(temp_bitboard, tf.cast(pos, dtype=tf.uint64)), 1)
            temp_feature_map = temp_feature_map.write(pos, tf.cast(bit, tf.float32))

        # reshape the 64 bitboard positions to 8x8 format, dimensions are (row, column)
        temp_feature_map = temp_feature_map.stack()
        temp_feature_map = tf.reshape(temp_feature_map, (8, 8))

        # apply the converted feature map to the output
        feature_maps = feature_maps.write(i, temp_feature_map)

    feature_maps = feature_maps.stack()
    return feature_maps


if __name__ == '__main__':

    # test single chessboard tensor conversion
    start_board = chesslib.ChessBoard_StartFormation()
    conv_start_board = chessboard_to_2d_feature_maps(start_board)
    print(conv_start_board.numpy())

    # test loading training and eval datasets
    batch_size = 32
    train_dataset, eval_dataset = load_datasets(batch_size)
    print(train_dataset)
    print(eval_dataset)

    # try to print the first batch item of each dataset
    train_iterator = iter(train_dataset)
    first_item = train_iterator.next()
    print(first_item)
    eval_iterator = iter(eval_dataset)
    first_item = eval_iterator.next()
    print(first_item)