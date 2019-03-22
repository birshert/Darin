import numpy as np
from copy import deepcopy
from Net import *
import torch
import torch.nn
import torch.nn.functional as F
import time


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
    def __init__(self, number, time):
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
        self.past4_black = deepcopy(self.empty)
        self.past4_white = deepcopy(self.empty)
        self.count_turns = 0

        self.model = Net()
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(
            torch.load("model_p{}.pth".format(number), map_location=lambda storage, loc: storage))
        self.model.eval()

        self.model2 = VNet()
        # self.model2 = torch.nn.DataParallel(self.model2)
        self.model2.load_state_dict(
            torch.load("model_v{}.pth".format(number), map_location=lambda storage, loc: storage))
        self.model2.eval()

    def get_pv(self, field):
        data_policy = deepcopy(
            np.stack((field[0], field[1], field[2], field[3], field[4], field[5], field[6], field[7], field[10])))

        data_value = deepcopy(field)

        data1 = torch.stack([torch.from_numpy(data_policy).type(torch.FloatTensor)])
        data2 = torch.stack([torch.from_numpy(data_value).type(torch.FloatTensor)])
        policy = self.model(data1)
        value = self.model2(data2)
        policy = F.softmax(policy, dim=1)
        value = F.softmax(value, dim=1)

        return policy.detach().numpy()[0], value.detach().numpy()[0]

    def move(self, field, turn):
        self.past4_black = deepcopy(self.past3_black)
        self.past4_white = deepcopy(self.past3_white)
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
                           self.past2_white, self.past3_black, self.past3_white, self.past4_black, self.past4_white,
                           turn_))

        policy, evaluation = self.get_pv(deepcopy(input_))

        possible = deepcopy(field.free)

        # move = policy.argmax()

        # while move not in possible:
        #     policy[move] = 0
        #     move = policy.argmax()
        # return move // 15, move % 15

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

        return move // 15, move % 15

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

        if evaluation[0] > evaluation[1]:
            if black:
                evaluation = -evaluation[0]
            else:
                evaluation = evaluation[0]
        else:
            if black:
                evaluation = evaluation[1]
            else:
                evaluation = -evaluation[1]

        if winner:
            evaluation = winner

        node = [0 for _ in range(225)]
        data[str(moves)] = [deepcopy(policy), deepcopy(node), deepcopy(node)]

        made_moves.reverse()

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
