from __future__ import print_function
import numpy as np
import torch
from copy import deepcopy


def parse(count1, count2):
    path = "train-1.renju"

    file = open(path, mode='r')

    change = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4,
              'e': 5,
              'f': 6,
              'g': 7,
              'h': 8,
              'j': 9,
              'k': 10,
              'l': 11,
              'm': 12,
              'n': 13,
              'o': 14,
              'p': 15}

    data_x = []
    data_y = []
    empty_field = np.array([[0.0 for _ in range(15)] for _ in range(15)])

    for pos, line in enumerate(file):
        if pos < count1:
            continue
        if pos > count2:
            break

        data = line.split()
        field = deepcopy(empty_field)
        if data[0] != 'draw':
            stone = 1.0
            for i in range(0, len(data) - 1):
                if i != 0:
                    move = [change[data[i][0]] - 1, int(data[i][1]) - 1]
                    field[move[0]][move[1]] = stone

                data_x.append(deepcopy(field))

                next_move = [change[data[i + 1][0]] - 1, int(data[i + 1][1]) - 1]

                data_y.append(deepcopy((next_move[0]) * 15 + next_move[1]))
                stone *= -1.0

    x = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in data_x])
    y = torch.stack([torch.tensor(i) for i in data_y])

    return x, y
