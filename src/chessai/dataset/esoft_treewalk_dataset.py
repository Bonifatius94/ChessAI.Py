
import numpy as np
import tensorflow as tf
import chesslib

from .chessboard_ext import conv_board


class ChessRandTreewalkDataset(object):

    def __init__(self, batch_size: int):

        super(ChessRandTreewalkDataset, self).__init__()
        self.batch_size = batch_size


    def load_datasets(self, num_episodes: int, policy: tf.keras.Model=None, epsilon: float=0.1):

        # do random chess game tree walks to generate fictional chess games
        # if specified, use the policy to generate games according to the policy

        # TODO: implement logic
        pass


    def generate_game(self, value_est: tf.keras.Model, epsilon: float=0.1):

        # load the start formation
        board = chesslib.ChessBoard_StartFormation()

        # intialize states and actions cache
        states = np.array([board])
        actions = np.array([])

        # perform actions until the game is over
        while chesslib.GameState(board) != chesslib.GameState_Checkmate \
            or chesslib.GameState(board) != chesslib.GameState_Tie:

            # get all possible actions
            poss_actions = chesslib.GenerateDraws(board)

            # get ratings for all possible actions
            next_states = [chesslib.ApplyDraw(draw) for draw in poss_actions]
            action_ratings = [value_est(conv_board(state)) for state in next_states]
            # state_rating = value_func(conv_board(board))
            # advantages = action_ratings - state_rating

            # choose the action to be applied (epsilon-greedy)
            explore = np.random.uniform() <= epsilon
            selected_action = np.random.choice(poss_actions) if explore \
                else poss_actions[np.argmin(action_ratings)]
            # note: argmin() is required because otherwise the AI would learn draws 
            #       that are beneficial for the opponent (argmax() -> maximize opp. reward?!)

            # write the (state, action) tuple to cache and update the board with the state
            new_state = chesslib.ApplyDraw(selected_action)
            states = np.append(states, np.array([new_state]))
            actions = np.append(actions, np.array([selected_action]))
            board = new_state

        

        # now that the game is over, bring the game data into SARS format
        rewards = []
        next_states = states[1:]
        states = states[:-2]

        return (states, actions, rewards, next_states)