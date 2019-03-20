from __future__ import print_function
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import TensorDataset
import gc


def random_shift(data):
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

    min_x = 15
    min_y = 15
    max_x = 0
    max_y = 0

    for i in range(1, len(data)):
        x, y = change[data[i][0]] - 1, int(data[i][1]) - 1
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return np.random.choice(np.arange(-min_x, 15 - max_x)), np.random.choice(np.arange(- min_y, 15 - max_y))


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
    white_turn = np.array([[0.0 for _ in range(15)] for _ in range(15)])
    black_turn = np.array([[1.0 for _ in range(15)] for _ in range(15)])
    empty_field = np.array([[0.0 for _ in range(15)] for _ in range(15)])

    for pos, line in enumerate(file):
        if pos < count1:
            continue
        if pos > count2:
            break

        data = line.split()
        shift_x, shift_y = random_shift(data)
        black_field = deepcopy(empty_field)
        white_field = deepcopy(empty_field)
        past1_white = deepcopy(empty_field)
        past1_black = deepcopy(empty_field)
        past2_white = deepcopy(empty_field)
        past2_black = deepcopy(empty_field)
        past3_black = deepcopy(empty_field)
        past3_white = deepcopy(empty_field)

        if data[0] != 'draw':
            black = False
            for i in range(0, len(data) - 1):
                if i > 3:
                    past3_black = deepcopy(past2_black)
                    past3_white = deepcopy(past2_white)

                if i > 2:
                    past2_white = deepcopy(past1_white)
                    past2_black = deepcopy(past1_black)

                if i > 1:
                    past1_white = deepcopy(white_field)
                    past1_black = deepcopy(black_field)

                if i > 0:
                    move = [change[data[i][0]] - 1 + shift_x, int(data[i][1]) - 1 + shift_y]
                    if black:
                        black_field[move[0]][move[1]] = 1.0
                    else:
                        white_field[move[0]][move[1]] = 1.0

                if not black:
                    if data[0] == 'black':
                        data_x.append(deepcopy(
                            np.stack(
                                (black_field, white_field, past1_black, past1_white, past2_black,
                                 past2_white, past3_black, past3_white, black_turn))))
                        next_move = [change[data[i + 1][0]] - 1, int(data[i + 1][1]) - 1]

                        data_y.append(deepcopy([((next_move[0] + shift_x) * 15 + next_move[1] + shift_y), 1.0]))
                else:
                    if data[0] == 'white':
                        data_x.append(deepcopy(
                            np.stack(
                                (black_field, white_field, past1_black, past1_white, past2_black,
                                 past2_white, past3_black, past3_white, white_turn))))
                        next_move = [change[data[i + 1][0]] - 1, int(data[i + 1][1]) - 1]

                        data_y.append(deepcopy([((next_move[0] + shift_x) * 15 + next_move[1] + shift_y), -1.0]))

                black = not black

    x = torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in data_x])
    y = torch.stack([torch.tensor(i) for i in data_y])

    del data_x
    del data_y

    gc.collect()

    return x, y


def make_dataset(count1, count2):
    x, y = parse(count1, count2)
    dataset = TensorDataset(x, y)
    del x, y
    gc.collect()
    return dataset
