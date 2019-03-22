import pygame
from MCTS import *
import numpy as np


class RandomPlayer:
    def __init__(self):
        pass

    @staticmethod
    def move_(field, turn):
        possible = field.free
        pos = np.random.randint(0, len(possible))
        return possible[pos] // 15, possible[pos] % 15


class HumanPlayer:
    def __init__(self):
        pass

    @staticmethod
    def move_(field, turn):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pos = [event.pos[0], event.pos[1]]
                        pos[0] -= 25
                        pos[1] -= 25
                        pos[0] //= 50
                        pos[1] //= 50
                        if field.get_node(pos[0], pos[1]).is_empty():
                            return pos
                        else:
                            print("INVALID MOVE\nCHOOSE ANOTHER\n")
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    exit()


class AI:
    def __init__(self, number):
        self.mcts = MCTS(number, 5)

    def move_(self, field, turn):
        ret, move = self.trick(deepcopy(field.get_black() * turn + field.get_white() * (not turn)), field.free)
        if ret:
            return move // 15, move % 15
        ret, move = self.trick(deepcopy(field.get_black() * (not turn) + field.get_white() * turn), field.free)
        if ret:
            return move // 15, move % 15

        return self.mcts.move(field, turn)

    def trick(self, board, free):
        for move in free:
            board[move // 15][move % 15] = 1.0
            if self.check_sequence(5, move, board):
                return True, move
            board[move // 15][move % 15] = 0.0
        return False, 0

    @staticmethod
    def check_sequence(n, move, board):
        i = move // 15
        j = move % 15

        # vertical check
        for shift in range(n):
            stones = []
            cur = 0
            for k in range(n):
                if 15 > i - k + shift >= 0 and 15 > j >= 0:
                    cur = board[i - k + shift][j]
                    if cur:
                        stones.append(cur)
            if len(stones) == n:
                return True

        # horizontal check
        for shift in range(n):
            stones = []
            cur = 0
            for k in range(n):
                if 15 > i >= 0 and 15 > j - k + shift >= 0:
                    cur = board[i][j - k + shift]
                    if cur:
                        stones.append(cur)
            if len(stones) == n:
                return True

        # diagonal check 1
        for shift in range(n):
            stones = []
            cur = 0
            for k in range(n):
                if 15 > i - k + shift >= 0 and 15 > j - k + shift >= 0:
                    cur = board[i - k + shift][j - k + shift]
                    if cur:
                        stones.append(cur)
            if len(stones) == n:
                return True

        # diagonal check 2
        for shift in range(n):
            stones = []
            cur = 0
            for k in range(n):
                if 15 > i - k + shift >= 0 and 15 > j + k - shift >= 0:
                    cur = board[i - k + shift][j + k - shift]
                    if cur:
                        stones.append(cur)
            if len(stones) == n:
                return True
        return False
