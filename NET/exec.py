import logging
import os
import random

import backend
import numpy as np
import renju
import torch
import torch.nn.functional as F
import time
from copy import deepcopy
import torch.nn as nn
import math


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

        self.free = [i for i in range(225)]

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
                    self.white[x][y] = 1.0
                self.field[x][y] = stone
                self.free.remove(x * 15 + y)

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(9, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.policy1 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self.policy2 = nn.Sequential(
            nn.Linear(450, 15 * 15),
            nn.ReLU(inplace=True)
        )

        self.weight_init(self.convolutional)
        self.weight_init(self.residual1)
        self.weight_init(self.residual2)
        self.weight_init(self.policy1)
        self.weight_init(self.policy2)

    def head(self, x):
        x = self.policy1(x)
        x = x.view(x.size(0), -1)
        x = self.policy2(x)

        return x

    @staticmethod
    def weight_init(elem):
        for m in elem.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.convolutional(x)  # convolutional

        out = self.residual1(x)  # residual tower
        x = x + out  #
        out = self.residual2(x)  #
        x = x + out  #

        return self.head(x)


class VNet(nn.Module):
    size = 15

    def __init__(self):
        super(VNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(32 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class KEYS:
    def __init__(self, elems=None):
        if elems is None:
            self.set1 = list()
            self.set2 = list()
        else:
            self.set1 = elems[0]
            self.set2 = elems[1]
            self.set = [self.set1, self.set2]
        self.normalize()

    def __len__(self):
        return len(self.set)

    def add(self, elem, pos):
        self.set[pos].append(elem)
        self.normalize()

    def normalize(self):
        self.set1 = sorted(set(self.set1))
        self.set2 = sorted(set(self.set2))
        self.set = [self.set1, self.set2]

    def __str__(self):
        return str(self.set)

    def remove(self, elem, pos):
        self.set[pos].remove(elem)
        self.normalize()

    def empty(self):
        return not (len(self.set1) or len(self.set2))


class MCTS:
    def __init__(self, time):
        self.iterations_time = time
        self.t = 1
        self.empty = np.array([[0.0 for _ in range(15)] for _ in range(15)])
        self.white_turn = np.array([[0.0 for _ in range(15)] for _ in range(15)])
        self.black_turn = np.array([[+1.0 for _ in range(15)] for _ in range(15)])
        self.black_field = deepcopy(self.empty)
        self.white_field = deepcopy(self.empty)
        self.past1_black = deepcopy(self.empty)
        self.past1_white = deepcopy(self.empty)
        self.past2_black = deepcopy(self.empty)
        self.past2_white = deepcopy(self.empty)
        self.past3_black = deepcopy(self.empty)
        self.past3_white = deepcopy(self.empty)
        self.count_turns = 0

        self.model = Net()
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(
            torch.load("model_p{}.pth".format(100), map_location=lambda storage, loc: storage))
        self.model.eval()

        self.model2 = VNet()
        self.model2 = torch.nn.DataParallel(self.model2)
        self.model2.load_state_dict(
            torch.load("model_v{}.pth".format(100), map_location=lambda storage, loc: storage))
        self.model2.eval()

    def get_pv(self, field):
        data_ = torch.stack([torch.from_numpy(field).type(torch.FloatTensor)])
        policy = self.model(data_)
        value = self.model2(data_)
        policy = F.softmax(policy, dim=1)
        value = F.softmax(value, dim=1)

        return policy.detach().numpy()[0], value.detach().numpy()[0]

    def move(self, field, turn):
        self.past3_black = deepcopy(self.past2_black)
        self.past3_white = deepcopy(self.past2_white)
        self.past2_black = deepcopy(self.past1_black)
        self.past2_white = deepcopy(self.past1_white)
        self.past1_black = deepcopy(self.black_field)
        self.past1_white = deepcopy(self.white_field)
        self.black_field = deepcopy(field.get_black())
        self.white_field = deepcopy(field.get_white())
        turn_ = self.black_turn * turn + (not turn) * self.white_turn

        input_ = np.stack((self.black_field, self.white_field, self.past1_black, self.past1_white, self.past2_black,
                           self.past2_white, self.past3_black, self.past3_white, turn_))

        policy, evaluation = self.get_pv(deepcopy(input_))

        possible = deepcopy(field.free)

        node = [0 for _ in range(225)]

        root = str(KEYS())

        data = {root: [policy, deepcopy(node), deepcopy(node)]}

        start = time.clock()
        eps = 0.2
        while time.clock() - start + eps < self.iterations_time:
            if len(possible) != 0:
                data = self.tree_search(data, deepcopy(possible), input_, deepcopy(turn))

        n_s = np.array(data[root][1])
        n_s = np.power(n_s, (1 / self.t)) / np.sum(np.power(n_s, (1 / self.t)))
        move = n_s.argmax()
        self.count_turns += 2
        if self.count_turns > 30:
            self.t = 0.5
        if self.count_turns > 50:
            self.t = 0.1

        return move

    @staticmethod
    def update_field(field, move):
        past3_black = deepcopy(field[4])
        past3_white = deepcopy(field[5])
        past2_black = deepcopy(field[2])
        past2_white = deepcopy(field[3])
        past1_black = deepcopy(field[0])
        past1_white = deepcopy(field[1])
        turn = deepcopy(field[2])
        if turn[0][0] > 0:
            field[0][move // 15][move % 15] = 1.0
        else:
            field[1][move // 15][move % 15] = 1.0
        field[2] = past1_black
        field[3] = past1_white
        field[4] = past2_black
        field[5] = past2_white
        field[6] = past3_black
        field[7] = past3_white
        field[8] = 1 - field[8]

        return field

    def tree_search(self, data, possible, field, turn):
        black = turn
        moves = KEYS()
        made_moves = []
        winner = 0

        while str(moves) in data.keys():
            if len(possible) == 0:
                break

            current = str(moves)
            policy = data[current][0]
            n_s = data[current][1]
            q = np.array(data[current][2])

            c = np.sqrt(np.sum(n_s) + 1)

            u = np.array([policy[i] * c / (n_s[i] + 1) for i in range(15 * 15)])

            choosing = 3 * u + q

            move = choosing.argmax()

            while move not in possible:
                choosing[move] -= max(choosing)
                move = choosing.argmax()

            possible.remove(move)
            moves.add(move, black)
            made_moves.append(move)
            field = self.update_field(deepcopy(field), move)

            black = not black

            if black:
                ret = self.check_sequence(5, move, field[0])
                if ret:
                    winner = 1
                    break
            else:
                ret = self.check_sequence(5, move, field[1])
                if ret:
                    winner = -1
                    break

        policy, evaluation = self.get_pv(field)
        if black:
            evaluation = evaluation[1]
        else:
            evaluation = -evaluation[0]

        if winner:
            evaluation = winner

        node = [0 for _ in range(225)]
        data[str(moves)] = [deepcopy(policy), deepcopy(node), deepcopy(node)]

        made_moves = list(reversed(made_moves))

        for move in made_moves:
            black = not black
            moves.remove(move, black)
            current = str(moves)
            n_s = data[current][1]
            q = data[current][2]
            n_s[move] += 1
            q[move] = (q[move] * (n_s[move] - 1) + evaluation) / n_s[move]
            data[current][1] = n_s
            data[current][2] = q

        return data

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


class AI_player():
    def __init__(self):
        self.mcts = MCTS(3)
        self.field = Field()
        self.black = True

    def move_(self, board):
        free = renju.list_positions(board, renju.Player.NONE)
        if len(free) == 15 * 15 - 1:
            self.black = False

        ret, move = self.trick(
            deepcopy(self.field.get_black() * self.black + self.field.get_white() * (not self.black)), self.field.free)
        if ret:
            return move // 15, move % 15
        ret, move = self.trick(
            deepcopy(self.field.get_black() * (not self.black) + self.field.get_white() * self.black), self.field.free)
        if ret:
            return move // 15, move % 15

        return self.changer(self.mcts.move(self.field, self.black))

    def board_to_field(self, board):
        pos_black = renju.list_positions(board, renju.Player.BLACK)
        pos_white = renju.list_positions(board, renju.Player.WHITE)
        turn = len(pos_white) == len(pos_black)

    def changer(self, move):
        letters = "abcdefghjklmnop"
        return letters[move // 15], move % 15 + 1

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


def main():
    pid = os.getpid()
    LOG_FORMAT = f'{pid}:%(levelname)s:%(asctime)s: %(message)s'

    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.debug("Start Lesha backend...")

    player = AI_player()

    try:
        while True:
            logging.debug("Wait for game update...")
            game = backend.wait_for_game_update()

            if not game:
                logging.debug("Game is over!")
                return

            logging.debug('Game: [%s]', game.dumps())
            move = player.move_(game.board())

            if not backend.set_move(move):
                logging.error("Impossible set move!")
                return

            logging.debug('Random move: %s', move)

    except:
        logging.error('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
