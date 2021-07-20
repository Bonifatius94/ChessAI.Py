
import chesslib
import numpy as np


class ChessGameEnv():
    """
    This is an implementation of a chess game environment (similar to OpenAI Gym).
    It serves following actions / use cases:

    1) store the current game state, last action applied, epiode and steps of episode counters
    2) retrieve all possible actions for the current game state (similar to OpenAI Gym's action space)
    3) simulate a n action to see the resulting game state (custom, due to determinism of chess)
    4) Apply an action to the current game environment (like OpenAI Gym)
    """

    def __init__(self, max_steps_per_episode: int=200):
        super(ChessGameEnv, self).__init__()

        self.max_steps_per_episode = max_steps_per_episode

        # initialize the episode and step of episode counters with zeros
        self.episode = 0
        self.episode_step = 0

        # initialize the game state with a chess board in start formation
        self.state = chesslib.ChessBoard_StartFormation()

        # initialize the last action with null
        self.last_action = chesslib.ChessDraw_Null


    def get_possible_actions(self) -> np.ndarray:

        # determine the drawing side (alternatingly white / black)
        drawing_side = self.episode_step % 2

        # generate all possible action for the given game state
        return chesslib.GenerateDraws(self.state,
            drawing_side, self.last_action, True)


    def simulate_action(self, action) -> np.ndarray:

        # simulate the action without actually
        # applying it to the game environment
        return chesslib.ApplyDraw(self.state, action)


    def step(self, action: int) -> tuple:

        # apply the action to the game state
        obs0 = self.state
        obs1 = chesslib.ApplyDraw(obs0, action)

        # update the game state (check for terminal)
        game_over_states = [chesslib.GameState_Checkmate, chesslib.GameState_Tie]
        game_state = chesslib.GameState(obs1, action)
        is_terminal = game_state in game_over_states \
            or self.episode_step >= self.max_steps_per_episode

        # compute the action's reward (win=1, tie=0.5, otherwise 0.0)
        is_win = game_state == chesslib.GameState_Checkmate
        is_tie = game_state == chesslib.GameState_Tie
        reward = 1.0 if is_win else (0.5 if is_tie else 0.0)

        # update episode and step per episode counters
        self.episode = self.episode + 1 if is_terminal else self.episode
        self.episode_step = self.episode_step + 1 if is_terminal else 0

        # apply the resulting game state (reset to start if terminal state occurs)
        self.state = obs1 if not is_terminal else chesslib.ChessBoard_StartFormation()

        # put the observation together
        return (obs1, reward, is_terminal)


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
