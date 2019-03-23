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
import torch.cuda
import math


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
            nn.LeakyReLU(inplace=True)
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
        x = self.convolutional(x)

        out = self.residual1(x)
        x = x + out
        out = self.residual2(x)
        x = x + out

        return self.head(x)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(11, 256, kernel_size=3, stride=1, padding=1),
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

        self.value1 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.value2 = nn.Sequential(
            nn.Linear(15 * 15, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh()
        )

        self.weight_init(self.convolutional)
        self.weight_init(self.residual1)
        self.weight_init(self.residual2)
        self.weight_init(self.value1)
        self.weight_init(self.value2)

    def head(self, x):
        x = self.value1(x)
        x = x.view(x.size(0), -1)
        x = self.value2(x)

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
        x = self.convolutional(x)
        out = self.residual1(x)
        x = x + out
        out = self.residual2(x)
        x = x + out

        return self.head(x)


class MCTS:
    def __init__(self, time):
        self.iterations_time = time
        self.t = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.black = True
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
        self.past4_black = deepcopy(self.empty)
        self.past4_white = deepcopy(self.empty)
        self.count_turns = 0

        self.model = Net()
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(
            torch.load("model_p{}.pth".format(100500), map_location=lambda storage, loc: storage))
        self.model.to(self.device)
        self.model.eval()

        self.model2 = VNet()
        self.model2.load_state_dict(
            torch.load("model_v{}.pth".format(100500), map_location=lambda storage, loc: storage))
        self.model2.to(self.device)
        self.model2.eval()

    def get_pv(self, field):
        data_policy = deepcopy(
            np.stack((field[0], field[1], field[2], field[3], field[4], field[5], field[6], field[7], field[10])))

        data_value = deepcopy(field)

        data1 = torch.stack([torch.from_numpy(data_policy).type(torch.FloatTensor).to(self.device)])
        data2 = torch.stack([torch.from_numpy(data_value).type(torch.FloatTensor).to(self.device)])
        policy = self.model(data1)
        value = self.model2(data2)
        policy = F.softmax(policy, dim=1)
        value = F.softmax(value, dim=1)

        return policy.detach().numpy()[0], value.detach().numpy()[0]

    def move(self, black, white, free, turn):
        self.black = turn
        self.past4_black = deepcopy(self.past3_black)
        self.past4_white = deepcopy(self.past3_white)
        self.past3_black = deepcopy(self.past2_black)
        self.past3_white = deepcopy(self.past2_white)
        self.past2_black = deepcopy(self.past1_black)
        self.past2_white = deepcopy(self.past1_white)
        self.past1_black = deepcopy(self.black_field)
        self.past1_white = deepcopy(self.white_field)
        self.black_field = deepcopy(black)
        self.white_field = deepcopy(white)
        turn_ = self.black_turn * turn + (not turn) * self.white_turn

        input_ = np.stack((self.black_field, self.white_field, self.past1_black, self.past1_white, self.past2_black,
                           self.past2_white, self.past3_black, self.past3_white, self.past4_black, self.past4_white,
                           turn_))

        policy, evaluation = self.get_pv(deepcopy(input_))

        possible = deepcopy(free)

        node = [0 for _ in range(225)]

        root = self.normalize([set(), set()])

        data = {root: [policy, deepcopy(node), deepcopy(node)]}

        start = time.clock()
        eps = 0.3
        while time.clock() - start + eps < self.iterations_time:
            if len(possible) != 0:
                data = self.tree_search(data, deepcopy(possible), input_)

        n_s = np.array(data[root][1])
        n_s = np.power(n_s, (1 / self.t)) / np.sum(np.power(n_s, (1 / self.t)))

        move = np.random.choice([i for i in range(225)], p=n_s)

        while move not in possible:
            n_s[move] -= 10
            move = np.random.choice([i for i in range(225)], p=n_s)

        self.count_turns += 2
        if self.count_turns > 30:
            self.t = 0.5
        if self.count_turns > 50:
            self.t = 0.1

        return [move // 15, move % 15]

    @staticmethod
    def normalize(key):
        return str(sorted(key[0])) + str(sorted(key[1]))

    @staticmethod
    def update_field(field, move):
        past4_black = deepcopy(field[6])
        past4_white = deepcopy(field[7])
        past3_black = deepcopy(field[4])
        past3_white = deepcopy(field[5])
        past2_black = deepcopy(field[2])
        past2_white = deepcopy(field[3])
        past1_black = deepcopy(field[0])
        past1_white = deepcopy(field[1])

        turn = deepcopy(field[10])
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
        field[8] = past4_black
        field[9] = past4_white
        field[10] = 1 - field[10]

        return field

    def tree_search(self, data, possible, field):
        black = deepcopy(self.black)
        moves = [set(), set()]
        made_moves = []
        winner = 0

        while self.normalize(moves) in data.keys():
            if len(possible) == 0:
                break

            current = self.normalize(moves)
            policy = data[current][0]
            n_s = data[current][1]
            q = np.array(data[current][2])

            c = np.sqrt(np.sum(n_s) + 1)

            u = np.array([policy[i] * c / (n_s[i] + 1) for i in range(15 * 15)])

            choosing = u + q

            move = choosing.argmax()

            while move not in possible:
                choosing[move] -= 100000
                move = choosing.argmax()

            possible.remove(move)
            moves[black].add(move)
            made_moves.append(move)
            field = self.update_field(deepcopy(field), move)

            black = not black

            if not black:
                if self.check_sequence(5, move, field[0]):
                    if self.black:
                        winner = 1
                    else:
                        winner = -1
                    break
            else:
                if self.check_sequence(5, move, field[1]):
                    if self.black:
                        winner = -1
                    else:
                        winner = 1
                    break

        policy, evaluation = self.get_pv(field)

        if not black:
            evaluation = (evaluation[1] - evaluation[0])
        else:
            evaluation = (evaluation[0] - evaluation[1])

        if winner:
            evaluation = winner

        node = [0 for _ in range(225)]
        data[self.normalize(moves)] = [deepcopy(policy), deepcopy(node), deepcopy(node)]

        made_moves.reverse()

        for move in made_moves:
            black = not black
            moves[black].remove(move)
            current = self.normalize(moves)
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
        self.black = True

    def move_(self, board):
        black, white = self.board_to_field(board)
        free = self.get_free(board)

        ret, move = self.trick(
            deepcopy(black * self.black + white * (not self.black)), free)
        if ret:
            return renju.to_move([move // 15, move % 15])

        ret, move = self.trick(
            deepcopy(black * (not self.black) + white * self.black), free)
        if ret:
            return renju.to_move([move // 15, move % 15])

        return renju.to_move(self.mcts.move(black, white, free, self.black))

    @staticmethod
    def get_free(board):
        free = renju.list_positions(board, renju.Player.NONE)
        ret = []
        for pos in free:
            ret.append(pos[0] * 15 + pos[1])
        return ret

    @staticmethod
    def board_to_field(board):
        pos_black = renju.list_positions(board, renju.Player.BLACK)
        pos_white = renju.list_positions(board, renju.Player.WHITE)
        black = [[0. for _ in range(15)] for _ in range(15)]
        white = [[0. for _ in range(15)] for _ in range(15)]
        for pos in pos_black:
            black[pos[0]][pos[1]] = 1.
        for pos in pos_white:
            white[pos[0]][pos[1]] = 1.

        return black, white

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
            field = game.board()
            player.black = True

            move = player.move_(field)

            if not backend.set_move(move):
                logging.error("Impossible set move!")
                return

            logging.debug('Random move: %s', move)

    except:
        logging.error('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
