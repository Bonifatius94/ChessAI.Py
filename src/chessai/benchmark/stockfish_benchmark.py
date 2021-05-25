
import chesslib
import numpy as np


class StockfishBenchmark():

    def __init__(self, stockfish_level: int):
        super(StockfishBenchmark, self).__init__()
        self.stockfish_level = stockfish_level


    def evaluate(self, num_games: int, choose_action_func) -> (float, float, float):

        # initialize the win/tie/loss counters with 0
        wins = 0
        ties = 0
        losses = 0

        # loop through all benchmark games
        for _ in range(num_games):

            # determine the side selection (white=0, black=1)
            stockfish_side = np.random.uniform(2)

            # play a single game to end
            drawing_side, game_state = self.play_single_game(
                choose_action_func, stockfish_side)

            # update the win/tie/loss counters according to the final game state
            if game_state == chesslib.GameState_Checkmate:
                stockfish_win = drawing_side != stockfish_side
                wins += 0 if stockfish_win else 1
                losses += 1 if stockfish_win else 0
            elif game_state == chesslib.GameState_Tie:
                ties += 1

        # return the win rate of the model evaluated against Stockfish
        return (wins / num_games, ties / num_games, losses / num_games)


    def play_single_game(self, choose_action_func, stockfish_side: int):

        # initialize the drawing side and last action caches
        drawing_side = chesslib.ChessColor_White
        last_action = chesslib.ChessDraw_Null

        # initialize the board with the start formation
        board = chesslib.ChessBoard_StartFormation()

        # set the initial game state and define the game-over states
        game_state = chesslib.GameState_None
        game_over_states = [chesslib.GameState_Checkmate, chesslib.GameState_Tie]

        # play until the game is over
        while game_state not in game_over_states:

            # let the drawing player choose his action
            draw_stockfish = drawing_side == stockfish_side
            action = self.stockfish_choose_action(board, last_action) if draw_stockfish \
                else choose_action_func(board, last_action)

            # apply the action to the board and update the game state
            board = chesslib.ApplyDraw(board, action)
            game_state = chesslib.GameState(board)

            # update the last action and drawing side caches
            last_action = action
            drawing_side = 1 - drawing_side

        return (drawing_side, game_state)


    def stockfish_choose_action(self, board, last_action):

        # TODO: implement logic
        # 1) convert the chessboard to a stockfish-compatible format
        # 2) let stockfish choose his draw
        # 3) convert the draw back to the chesslib draw format and return it
        pass
