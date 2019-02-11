import numpy as np


class RandomPlayer:
    def move_(self, possible_moves):
        pos = np.random.randint(0, len(possible_moves))
        return possible_moves[pos]
