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

    x = []
    y = []
    black = torch.tensor(1)
    white = torch.tensor(-1)
    black_ = np.array([[1 for _ in range(15)] for _ in range(15)])
    white_ = np.array([[-1 for _ in range(15)] for _ in range(15)])
    empty = torch.from_numpy(np.array([0 for _ in range(15 * 15)])).type(torch.FloatTensor)

    for pos, line in enumerate(file):
        if pos < count1:
            continue
        if pos > count2:
            break

        data = line.split()
        black_field = np.array([[0 for _ in range(15)] for _ in range(15)])
        white_field = np.array([[0 for _ in range(15)] for _ in range(15)])
        if data[0] == 'black':
            turn_black = True
            for i in range(1, len(data) - 1):
                move = [change[data[i][0]], int(data[i][1])]

                next_move = [change[data[i + 1][0]], int(data[i + 1][1])]
                move_field = deepcopy(empty)
                move_field[(next_move[0] - 1) * 15 + (next_move[1] - 1)] = 1.0

                if turn_black:
                    black_field[move[0] - 1][move[1] - 1] = black
                else:
                    white_field[move[0] - 1][move[1] - 1] = white

                turn_black = not turn_black
                turn = turn_black * black_ + (not turn_black) * white_
                x.append(torch.from_numpy(np.stack((black_field, white_field, turn), axis=-1)))
                y.append([black, move_field])

        elif data[0] == 'white':
            turn_black = True
            for i in range(1, len(data) - 1):
                move = [change[data[i][0]], int(data[i][1])]

                next_move = [change[data[i + 1][0]], int(data[i + 1][1])]
                move_field = deepcopy(empty)
                move_field[(next_move[0] - 1) * 15 + (next_move[1] - 1)] = 1.0

                if turn_black:
                    black_field[move[0] - 1][move[1] - 1] = black
                else:
                    white_field[move[0] - 1][move[1] - 1] = white

                turn_black = not turn_black
                turn = turn_black * black_ + (not turn_black) * white_
                x.append(torch.from_numpy(np.stack((black_field, white_field, turn), axis=-1)))
                y.append([white, move_field])

    return x, y
