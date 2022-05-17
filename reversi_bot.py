import numpy as np
import math

class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.max_depth = 4

    def make_move(self, state) -> list:
        '''
        This is the only function that needs to be implemented for the lab!
        The bot should take a game state and return a move.

        The parameter "state" is of type ReversiGameState and has two useful
        member variables. The first is "board", which is an 8x8 numpy array
        of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
        there is a 1 that means the spot has one of player 1's stones. If
        there is a 2 on the spot that means that spot has one of player 2's
        stones. The other useful member variable is "turn", which is 1 if it's
        player 1's turn and 2 if it's player 2's turn.

        ReversiGameState objects have a nice method called get_valid_moves.
        When you invoke it on a ReversiGameState object a list of valid
        moves for that state is returned in the form of a list of tuples.

        Move should be a tuple (row, col) of the move you want the bot to make.
        '''
        valid_moves = state.get_valid_moves()
        if len(valid_moves) == 0: return None

        best_score = -math.inf
        best_move = []
        for move in valid_moves:
            val = self.maximize(self.hypothetically_move(state, move, self.move_num), math.inf, 0)
            if val > best_score:
                best_score = val
                best_move = move

        return best_move

    def maximize(self, state, high_bound: int, curr_depth: int):
        if curr_depth == self.max_depth: return self.heuristic(state)
        max = -math.inf
        for move in state.get_valid_moves():
            child_value = self.minimize(self.hypothetically_move(state, move, self.move_num), max, curr_depth + 1)
            if child_value > high_bound: return math.inf
            if child_value > max: max = child_value
        return max

    def minimize(self, state, low_bound: int, curr_depth: int):
        if curr_depth == self.max_depth: return self.heuristic(state)
        min = math.inf
        for move in state.get_valid_moves():
            child_value = self.maximize(self.hypothetically_move(state, move, 3 - self.move_num), min, curr_depth + 1)
            if child_value < low_bound: return -math.inf
            if child_value < min: min = child_value
        return min

    def heuristic(self, state) -> int:
        my_points = 0
        enemy_pionts = 0
        for row in state.board:
            for col in row:
                if col ==  0:
                    continue
                if col == self.move_num:
                    my_points += 1
                else:
                    enemy_pionts += 1
        return my_points - enemy_pionts

    def hypothetically_move(self, state, move: list, turn: int):
        '''Return the state that would result from making a move'''
        to_flip = []
        r, c = move[0], move[1]
        move = np.array(move)
        board = []
        for i in range(len(state.board)):
            row = state.board[i]
            board.append([])
            for j in range(len(row)):
                board[i].append(state.board[i][j])

        dim = state.board_dim
        board[r][c] = turn
        
        directions = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
        for item in directions:
            direction = np.array(item)
            to_check = move + direction
            could_flip = []
            while self.is_in_bounds(dim, to_check) and board[to_check[0]][to_check[1]] == 3 - turn:
                could_flip.append(list(to_check))
                to_check = to_check + direction
            if self.is_in_bounds(dim, to_check) and board[to_check[0]][to_check[1]] == turn:
                for coin in could_flip:
                    to_flip.append(coin)
        
        for coin in to_flip:
            board[coin[0]][coin[1]] = turn

        return ReversiGameState(np.array(board), 3 - turn)

    def is_in_bounds(self, dim, move):
        return move[0] >= 0 and move[0] < dim and move[1] >= 0 and move[1] < dim


class ReversiGameState:
    def __init__(self, board, turn):
        self.board_dim = 8  # Reversi is played on an 8x8 board
        self.board = board
        self.turn = turn  # Whose turn is it

    def capture_will_occur(self, row, col, xdir, ydir, could_capture=0):
        # We shouldn't be able to leave the board
        if not self.space_is_on_board(row, col):
            return False

        # If we're on a space associated with our turn and we have pieces
        # that could be captured return True. If there are no pieces that
        # could be captured that means we have consecutive bot pieces.
        if self.board[row, col] == self.turn:
            return could_capture != 0

        if self.space_is_unoccupied(row, col):
            return False

        return self.capture_will_occur(row + ydir,
                                       col + xdir,
                                       xdir, ydir,
                                       could_capture + 1)

    def space_is_on_board(self, row, col):
        return 0 <= row < self.board_dim and 0 <= col < self.board_dim

    def space_is_unoccupied(self, row, col):
        return self.board[row, col] == 0

    def space_is_available(self, row, col):
        return self.space_is_on_board(row, col) and \
            self.space_is_unoccupied(row, col)

    def is_valid_move(self, row, col):
        if self.space_is_available(row, col):
            # A valid move results in capture
            for xdir in range(-1, 2):
                for ydir in range(-1, 2):
                    if xdir == ydir == 0:
                        continue
                    if self.capture_will_occur(row + ydir, col + xdir, xdir, ydir):
                        return True

    def get_valid_moves(self):
        valid_moves = []

        # If the middle four squares aren't taken the remaining ones are all
        # that is available
        if 0 in self.board[3:5, 3:5]:
            for row in range(3, 5):
                for col in range(3, 5):
                    if self.board[row, col] == 0:
                        valid_moves.append((row, col))
        else:
            for row in range(self.board_dim):
                for col in range(self.board_dim):
                    if self.is_valid_move(row, col):
                        valid_moves.append((row, col))

        return valid_moves
