import numpy as np
from copy import deepcopy


class Node:
    def __init__(self, empty=True, stone=None):
        self.empty = empty
        if not empty:
            self.stone = stone
        else:
            self.stone = None

    def set_stone(self, stone):
        self.empty = False
        self.stone = stone

    def get_stone(self):
        return self.stone

    def is_empty(self):
        return self.empty

    def color(self):
        if self.stone is not None:
            return (self.stone == -1) * (255, 255, 255) + (self.stone == 1) * (0, 0, 0)


class Field:
    def __init__(self, start=None):
        if start is None:
            start = [[Node() for _ in range(15)] for _ in range(15)]

        self.start = start
        self.size = 15
        self.data = start
        self.field = np.array([[0.0 for _ in range(15)] for _ in range(15)])
        self.white = [[0.0 for _ in range(15)] for _ in range(15)]
        self.black = [[0.0 for _ in range(15)] for _ in range(15)]
        for i in range(15):
            for j in range(15):
                if not start[i][j].is_empty():
                    if start[i][j].get_stone() == 1:
                        self.black[i][j] = 1
                    elif start[i][j].get_stone() == -1:
                        self.white[i][j] = -1

    def get_size(self):
        return self.size

    def get_node(self, x, y):
        if not (x < 0 or x > (self.size - 1) or y < 0 or y > (self.size - 1)):
            return self.data[x][y]

    def make_move(self, x, y, stone):
        if not (x < 0 or x > self.size or y < 0 or y > self.size - 1):
            if not self.data[x][y].is_empty():
                return 0
            else:
                self.data[x][y].set_stone(stone)
                if stone == 1:
                    self.black[x][y] = 1.0
                elif stone == -1:
                    self.white[x][y] = -1.0
                self.field[x][y] = stone

    def get_field(self):
        return deepcopy(self.data)

    def field_(self):
        return deepcopy(self.field)

    def reset(self):
        self.data = self.start

    def get_white(self):
        return deepcopy(self.white)

    def get_black(self):
        return deepcopy(self.black)
