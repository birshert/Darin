import numpy as np


class RandomPlayer:
    def move_(self, possible_moves):
        return np.random.choice(possible_moves)
