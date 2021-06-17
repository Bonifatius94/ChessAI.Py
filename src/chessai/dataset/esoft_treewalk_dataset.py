
import numpy as np
import tensorflow as tf
import chesslib

from .dataset_utils import convert_states

# TODO: finish the implementation of this dataset generator class
#       it should create self-play chess data exploring the game tree
#       with a given exploration rate epsilon


class ChessESoftTreewalkDataset(object):

    def __init__(self):
        super(ChessESoftTreewalkDataset, self).__init__()


    def load_dataset(self, value_est: tf.keras.Model, epsilon: float=0.1, 
            batch_size: int=32, shuffle: bool=True):

        self.value_est = value_est

        # do epsilon-soft chess game tree walks to generate chess games
        # by self-play using the given value estimator (to be trained)
        generator = lambda eps: self.sars_selfplay_generator(eps)
        out_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32)

        # create a tensorflow dataset using the generator
        dataset = tf.data.Dataset.from_generator(generator, out_types, args=[epsilon])

        # batch and shuffle the dataset properly
        dataset = dataset.batch(batch_size)
        if shuffle: dataset = dataset.shuffle(50)

        # TODO: handle memory leaks of infinite datasets

        return dataset


    def sars_selfplay_generator(self, epsilon: float=0.1):

        # create an infinite amount of games
        while True:

            actions_count = 0

            # load the start formation
            obs0 = chesslib.ChessBoard_StartFormation()
            conv_obs0 = np.squeeze(convert_states(np.expand_dims(obs0, 0)))

            # define game over states
            game_over_states = [chesslib.GameState_Checkmate, chesslib.GameState_Tie]

            # initialize the cache variables game state, last action, etc.
            game_state = chesslib.GameState_None
            drawing_side = chesslib.ChessColor_White
            last_action = chesslib.ChessDraw_Null
            is_terminal = False

            # perform actions until the game is over
            while not is_terminal and actions_count <= 100:

                # get all possible actions
                poss_actions = chesslib.GenerateDraws(obs0, drawing_side, last_action, True)
                num_actions = len(poss_actions)

                # get ratings for all possible actions
                poss_obs1 = [chesslib.ApplyDraw(obs0, action) for action in poss_actions]
                conv_obs1 = convert_states(np.reshape(poss_obs1, (num_actions, 13)))
                action_ratings = self.value_est(conv_obs1).numpy() * -1
                # note: * -1 is required to avoid learning actions beneficial for the opponent

                # choose the action to be applied (epsilon-greedy)
                explore = np.random.uniform() <= epsilon
                best_action_id = np.argmax(action_ratings)
                action_id = np.random.randint(num_actions) if explore else best_action_id
                selected_action = poss_actions[action_id]
                obs1 = poss_obs1[action_id]

                # update the game state (check for terminal)
                game_state = chesslib.GameState(obs1, selected_action)
                is_terminal = game_state in game_over_states

                # compute the action's reward (win=1, tie=0.5, otherwise 0.0)
                is_win = game_state == chesslib.GameState_Checkmate
                is_tie = game_state == chesslib.GameState_Tie
                reward = 1.0 if is_win else (0.5 if is_tie else 0.0)

                # yield the current game timestep
                yield (conv_obs0, selected_action, reward, conv_obs1[action_id], is_terminal)
                # yield (conv_obs0, np.array([selected_action]), np.array([reward]),
                #        conv_obs1[action_id], np.array([is_terminal]))

                # update the cache variables
                obs0 = obs1
                conv_obs0 = conv_obs1[action_id]
                last_action = selected_action
                drawing_side = 1 - drawing_side
                actions_count += 1

                # # cache the selected actions and abort if there is a draw repetition
                # all_game_actions = np.append(all_game_actions, [selected_action])
                # if self.has_draw_repetition(all_game_actions): break


    # def has_draw_repetition(self, draw_history: np.ndarray):

    #     # minimal loop in chess is >= 4, both players need to draw and revert
    #     min_loop_size = 4
    #     max_loop_size = int(np.floor(len(draw_history) / 2))

    #     # loop through search size
    #     for loop_size in range(min_loop_size, max_loop_size):

    #         # split the data in the middle
    #         rest_draws = draw_history[:-loop_size]
    #         loop_draws = draw_history[-loop_size:]

    #         # loop through offsets
    #         for diff in range(len(rest_draws) - len(loop_draws) + 1):

    #             # check for sequences equal to the loop draws
    #             if np.all(rest_draws[diff:diff+loop_size] == loop_draws): return True

    #     return False
