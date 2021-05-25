
import sqlite3
import requests
import os, sys

import numpy as np
import pandas as pd
import tensorflow as tf
import chesslib


class ChessGmGamesDataset(object):

    def __init__(self, batch_size: int):

        super(ChessGmGamesDataset, self).__init__()
        self.batch_size = batch_size


    def load_datasets(self, sample_seed: int=None) -> (tf.data.Dataset, tf.data.Dataset):

        # make sure the winrates database is locally available (download if not)
        db_filepath = './win_rates.db'
        self.download_winrates_db(db_filepath)

        # load the training data into a pandas dataframe using a SQL query
        winrates_cache = self.load_pandas_winrates(db_filepath)

        # split the data into randomly sampled training and evaluation chunks (9:1)
        train_data = winrates_cache.sample(frac=0.9, random_state=sample_seed)
        eval_data = winrates_cache.drop(train_data.index)

        # convert the data into the SARS format
        train_data = self.prepare_pandas_data(train_data)
        eval_data = self.prepare_pandas_data(eval_data)

        # transform the pandas dataframe into a tensorflow dataset
        train_dataset = self.create_tf_dataset(train_data, self.batch_size, train=True)
        eval_dataset = self.create_tf_dataset(eval_data, self.batch_size, train=False)

        return (train_dataset, eval_dataset)


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
        states = np.array([np.array(self.bitboards_from_hash(x)) for x in pd_winrates['BoardBeforeHash']])
        actions = np.array(pd_winrates['DrawHashNumeric'])
        rewards = np.array(pd_winrates['WinRate'])
        next_states = np.array([chesslib.ApplyDraw(states[i], actions[i]) for i in range(len(states))])

        print(states.shape, actions.shape, rewards.shape, next_states.shape)

        # convert the states into the 2D format (7 channels) used for feature extraction
        conv_states = self.convert_states(states)
        conv_next_states = self.convert_states(next_states)

        return (conv_states, actions, rewards, conv_next_states)


    def bitboards_from_hash(self, board_hash: str):
        return chesslib.Board_FromHash(np.frombuffer(bytes.fromhex(board_hash), dtype=np.uint8))


    # def compact_draw_from_hash(self, draw_hash: str):
    #     return int(draw_hash) & 0x7FFF


    def create_tf_dataset(self, sars_data: tuple, batch_size: int, train: bool):

        # unwrap SARS data
        states, actions, rewards, next_states = sars_data

        # convert the pandas dataframe's columns to tensor slices
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # create a dataset from the tensor slices -> SARS tuples
        dataset = tf.data.Dataset.from_tensor_slices(
            tensors=(states, actions, rewards, next_states))

        # batch the data properly
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # shuffle the dataset when creating a training dataset
        if train: dataset = dataset.shuffle(50)

        return dataset
