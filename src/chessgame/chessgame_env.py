
# import standard libs
import math
import numpy as np


# import gym and chess framework
import gym
from gym import error, spaces
import chesslib


# init logger
import logging
logger = logging.getLogger(__name__)


# declare the gym environment implementation for playing chess games
class ChessGame(gym.Env):


    # init variables
    board = chesslib.ChessBoard_StartFormation
    draw_history = []
    drawing_side = 0


    def __init__(self):

        # init action and observation space
        action_max = 33554432
        max_bitboard = np.iinfo(np.uint64).max
        observation_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        observation_max = np.array([
            max_bitboard, max_bitboard, max_bitboard, max_bitboard, max_bitboard, 
            max_bitboard, max_bitboard, max_bitboard, max_bitboard, max_bitboard, 
            max_bitboard, max_bitboard, max_bitboard, action_max])

        self.action_space = spaces.Discrete(action_max)
        self.observation_space = spaces.Box(observation_min, observation_max)


    def _step(self, action):

        self._take_action(action)
        reward = self._get_reward()
        observation = np.append(self.board, action)
        is_game_over = reward != 0

        return observation, reward, is_game_over, { 'draws': self.draw_history }


    def _take_action(self, action):

        # apply the chess draw from action
        # assume that the agent only puts valid draws
        self.board = chesslib.ApplyDraw(self.board, action)
        self.draw_history.append(action)
        self.drawing_side = (self.drawing_side + 1) % 2


    def _reset(self):

        # reset the chess board to start formation
        # and assign the draw history with an empty list
        self.board = chesslib.ChessBoard_StartFormation
        self.draw_history = self.draw_history
        self.drawing_side = 0


    def _get_reward(self):

        # return following rewards according to the chess game's outcome
        # as soon as the game is considered over:
        #    victory -> +1
        #    defeat  -> -1
        #    tie     -> +0

        last_draw = self.draw_history[-2] if len(self.draw_history) >= 2 else cheslib.ChessDraw_Null
        step_draw = self.draw_history[-1]
        state = chesslib.GameState(self.board, last_draw)
        enemy_draws = chesslib.GenerateDraws(self.board, self.drawing_side, step_draw, True)

        if state == chesslib.GameSate_Checkmate:
            return -1
        elif state == chesslib.GameState_Tie:
            return 0.25
        elif len(enemy_draws) == 0:
            return 1
        else:
            return 0


    # info: no visualization required yet, may come later
    def _render(self, mode='human', close=False):
        # TODO: do just a very simple text output
        pass