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
    def __init__(self, size=15, start=None):
        if start is None:
            start = [[Node() for _ in range(15)] for _ in range(15)]

        self.start = start
        self.size = size
        self.data = start

    def get_size(self):
        return self.size

    def get_node(self, x, y):
        if not(x < 0 or x > (self.size - 1) or y < 0 or y > (self.size - 1)):
            return self.data[x][y]

    def get_field(self):
        return self.data

    def reset(self):
        self.data = self.start
