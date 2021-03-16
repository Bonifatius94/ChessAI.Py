
import sqlite3
import requests
import os, sys

import numpy as np
import pandas as pd
import tensorflow as tf
import chesslib

from .chessboard_ext import chessboard_to_compact_2d_feature_maps as conv_board


class ChessGmGamesDataset(object):

    def __init__(self, batch_size: int):

        super(ChessGmGamesDataset, self).__init__()
        self.batch_size = batch_size


    def load_datasets(self):

        # make sure the winrates database is locally available (download if not)
        db_filepath = './win_rates.db'
        self.download_winrates_db(db_filepath)

        # load the training data into a pandas dataframe using a SQL query
        # then, convert the dataframe to SARS format
        winrates_cache = self.load_pandas_winrates(db_filepath)
        sars_winrates = self.prepare_pandas_data(winrates_cache)

        # split the data into randomly sampled training and evaluation chunks (9:1)
        train_data = sars_winrates.sample(frac=0.9)
        eval_data = sars_winrates.drop(train_data.index)

        # transform the pandas dataframe into a tensorflow dataset
        train_dataset = self.create_tf_dataset(train_data, self.batch_size, train=True)
        eval_dataset = self.create_tf_dataset(eval_data, self.batch_size, train=False)

        return train_dataset, eval_dataset


    def download_winrates_db(self, db_filepath: str):

        # only download the db file if it does not already exist
        if not os.path.exists(db_filepath) or not os.path.isfile(db_filepath):

            # download the sqlite database file and write it to the given local filepath
            url = 'https://raw.githubusercontent.com/Bonifatius94/ChessAI.CS/master/Chess.AI/Data/win_rates.db'
            req = requests.get(url, allow_redirects=True)
            with open(db_filepath, 'wb') as out_file:
                out_file.write(req.content)


    def load_pandas_winrates(self, db_filepath: str):

        conn = sqlite3.connect(db_filepath)
        sql_str = 'SELECT BoardBeforeHash, DrawHashNumeric, WinRate \
            FROM WinRateInfo -- WHERE AnalyzedGames >= 10'
        win_rates_cache = pd.read_sql(sql_str, conn)
        conn.close()
        return win_rates_cache


    def prepare_pandas_data(self, pd_winrates: pd.DataFrame):

        # convert the dataframe into SARS data slices (state-action-reward-nextstate)
        states = [self.bitboards_from_hash(x) for x in pd_winrates['BoardBeforeHash']]
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


    def bitboards_from_hash(self, board_hash: str):
        return chesslib.Board_FromHash(np.frombuffer(bytes.fromhex(board_hash), dtype=np.uint8))


    def compact_draw_from_hash(self, draw_hash: str):
        return int(draw_hash) & 0x7FFF


    def create_tf_dataset(self, pd_sars_winrates: pd.DataFrame, batch_size: int, train: bool):

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
        dataset = dataset.map(lambda state, action, reward, next_state:
            (conv_board(state), action, reward, conv_board(next_state)),
            num_parallel_calls=8, deterministic=True
        )

        # batch the data properly
        dataset = dataset.batch(batch_size)

        # shuffle the dataset when creating a training dataset
        if train: dataset = dataset.shuffle(50)

        return dataset


# this is just a test script making sure all functions work fine
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