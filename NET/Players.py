import numpy as np


class RandomPlayer:
    def __init__(self):
        self.possible_ = []

    def possible_moves(self, field):
        moves = []
        for i in range(field.get_size()):
            for j in range(field.get_size()):
                if field.get_node(i, j).is_empty():
                    moves.append([i, j])
        self.possible_ = moves

    def move_(self, field):
        self.possible_moves(field)
        pos = np.random.randint(0, len(self.possible_))
        return self.possible_[pos]
