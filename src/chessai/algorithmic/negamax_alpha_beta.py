
import numpy as np
import multiprocessing as mp
import datetime as dt
import time
import chesslib


class AlgorithmicModel:

    def __init__(self):

        # initialize chess score cache
        # this cache stores (board_hash, score) tuples as a dictionary
        self.cache = dict()


    def __call__(self, chessboard: np.ndarray, valid_draws: np.ndarray, last_draw: int, comp_time: int):
        """
        Determine the best draw from the list for the given chess position.
        Therefore use the negamax algorithm with alpha-beta prune and iterative deepening.

        :param chessboard: The chess board representing the current position (bitboard format).
        :param valid_draws: The valid draws to be evaluated.
        :param comp_time: The max. amount of time to compute the result (in seconds).
        :returns: The best draw on the valid_draws list.
        """

        start_timestamp = dt.datetime.utcnow()

        # initialize all draws with neutral scores (represent the draws with the resulting chess position)
        next_boards = np.array([chesslib.ApplyDraw(chessboard, draw) for draw in valid_draws])
        next_board_hashes = np.array([self.board_to_str(board) for board in next_boards])
        est_scores = np.array([self.cache[hash] if hash in self.cache else 0.0 for hash in next_board_hashes])
        print("est. scores:", est_scores)

        # run iterative deepening as deep as possible within the given time limit
        depth = 1
        while depth <= 10:

            _ = self.negamax(chessboard, valid_draws, int(last_draw), depth, float('-inf'), float('inf'))

            # update the scores (will be used when the computation time is up)
            est_scores = np.array([self.cache[hash] if hash in self.cache else 0.0 for hash in next_board_hashes])
            print("depth", depth, "est. scores:", est_scores)

            depth += 1

        # determine the draw with the highest score
        return [(valid_draws[i], est_scores[i]) for i in range(len(valid_draws))]


    def term_process(self, timeout_secs: float, process_to_kill: mp.Process):

        time.sleep(timeout_secs)
        process_to_kill.terminate()


    def negamax(self, bitboards: np.ndarray, valid_draws: np.ndarray, last_draw,
                depth: int, alpha: float, beta: float) -> float:

        # print('negamax depth:', depth)

        # determine the drawing side
        drawing_side = valid_draws[0] >> 23 & 1 if len(valid_draws) > 0 else 0
        # print('drawing side:', drawing_side)

        # handle recursion termination cases (max. search depth or terminal state reached)
        state = chesslib.GameState(bitboards, last_draw)
        if state == chesslib.GameState_Checkmate: return float('-inf')
        # elif state == chesslib.GameState_Tie: return 0.0
        elif depth == 0: return self.eval_chessboard(bitboards, drawing_side)

        # determine the estimated scores for each draw
        # then, order the draws by their estimated score descendingly
        # this ensures trying out promising draws first and achieving more cut-offs
        next_boards = np.array([chesslib.ApplyDraw(bitboards, draw) for draw in valid_draws])
        next_board_hashes = np.array([self.board_to_str(board) for board in next_boards])
        est_scores = np.array([self.cache[hash] if hash in self.cache else 0.0 for hash in next_board_hashes])
        sort_perm = np.argsort(est_scores * -1.0) # sort by descending scores

        # initialize the estimated value as negative infinity
        value = float('-inf')

        # try out the draws one-by-one and estimate their outcomes recursively
        # therefore process the most promising draws first according to the score permuation
        for i in sort_perm:

            draw = int(valid_draws[i])
            next_board = next_boards[i]

            # compute the resulting chess board and the successing valid draws to be tried out
            # print('next draws:', next_board, drawing_side, draw)
            next_valid_draws = chesslib.GenerateDraws(next_board, int(drawing_side), int(draw), True)

            # perform a recursive function call to estimate the goodness of the draw
            est_draw_score = self.negamax(next_board, next_valid_draws, draw, depth - 1, -alpha, -beta)

            # update the cache with the new estimated score
            # TODO: make sure that the score does not need to be inverted first
            self.cache[next_board_hashes[i]] = est_draw_score

            # update alpha and beta
            value = max(value, est_draw_score * -1.0)
            alpha = max(alpha, value)

            # perform cut-off (there is a good enemy reply -> stop computing!)
            if alpha >= beta: break

        # return the estimated value of the given chess position considering only best draws
        return value


    def eval_chessboard(self, bitboards: np.ndarray, drawing_side: int):

        # define the factors for each bitboard by piece type
        piece_factors = np.array([
            200.0, 9.0, 5.0, 3.0, 3.0, 1.0,       # white pieces (king, queen, rook, bishop, knight, pawn)
            -200.0, -9.0, -5.0, -3.0, -3.0, -1.0, # black pieces (king, queen, rook, bishop, knight, pawn)
            0                                     # was_moved map (can be ignored)
        ])

        # get the piece counts for each side and piece type
        piece_counts = np.array([self.set_bits_count(bitboard) for bitboard in bitboards])

        # multiply the piece counts with the piece type factors and summarize everything
        score = np.sum(piece_factors * piece_counts)

        # invert the score for black side by multiplying with -1
        score *= 1 if drawing_side == chesslib.ChessColor_White else -1

        return score


    def set_bits_count(self, n: int):

        # snippet source: https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer

        temp = np.uint64(n)
        temp = (temp & np.uint64(0x5555555555555555)) + ((temp & np.uint64(0xAAAAAAAAAAAAAAAA)) >> np.uint64(1))
        temp = (temp & np.uint64(0x3333333333333333)) + ((temp & np.uint64(0xCCCCCCCCCCCCCCCC)) >> np.uint64(2))
        temp = (temp & np.uint64(0x0F0F0F0F0F0F0F0F)) + ((temp & np.uint64(0xF0F0F0F0F0F0F0F0)) >> np.uint64(4))
        temp = (temp & np.uint64(0x00FF00FF00FF00FF)) + ((temp & np.uint64(0xFF00FF00FF00FF00)) >> np.uint64(8))
        temp = (temp & np.uint64(0x0000FFFF0000FFFF)) + ((temp & np.uint64(0xFFFF0000FFFF0000)) >> np.uint64(16))
        temp = (temp & np.uint64(0x00000000FFFFFFFF)) + ((temp & np.uint64(0xFFFFFFFF00000000)) >> np.uint64(32))
        return temp


    def board_to_str(self, chessboard: np.ndarray):
        return str(bytes.hex(bytes(chesslib.Board_ToHash(chessboard))))


if __name__ == '__main__':

    # create an algorithmic model instance
    model = AlgorithmicModel()
    comp_time = 10 # 10 seconds of computation time to find out the best draw
    epsilon = 0.1

    # play some games until the end
    for i in range(1000):

        # initialize the board with the start formation and white side drawing
        drawing_side = chesslib.ChessColor_White
        last_draw = chesslib.ChessDraw_Null
        board = chesslib.ChessBoard_StartFormation()
        print(chesslib.VisualizeBoard(board))

        while True:

            # determine the estimated draw scores
            draws = chesslib.GenerateDraws(board, drawing_side, int(last_draw), True)
            draws_x_scores = model(board, draws, last_draw, comp_time)

            # determine the best and the second best draw
            draws_desc_by_score = sorted(draws_x_scores, key=1)
            best_draw = draws_desc_by_score[0][0]
            second_best_draw = draws_desc_by_score[1][0] if len(draws) > 1 else best_draw

            # sometimes, use only the second best draw for a bit of variety
            draw_to_apply = best_draw if epsilon < np.random.uniform(1) else second_best_draw

            board = chesslib.ApplyDraw(board, draw_to_apply)
            print(chesslib.VisualizeBoard(board))

            # check whether the game is over (finalize the game in case)
            last_draw = draw_to_apply
            state = chesslib.GameState(board, last_draw)
            if state == chesslib.GameState_Checkmate or state == chesslib.GameState_Tie: break
