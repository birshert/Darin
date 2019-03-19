import numpy as np
from copy import deepcopy
from Net import *
import torch
import torch.nn
import torch.nn.functional as F


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


class MCTS:
    def __init__(self, number, iterations):
        self.iterations = iterations
        self.empty = np.array([[0.0 for _ in range(15)] for _ in range(15)])
        self.white_turn = np.array([[-1.0 for _ in range(15)] for _ in range(15)])
        self.black_turn = np.array([[+1.0 for _ in range(15)] for _ in range(15)])
        self.black_field = deepcopy(self.empty)
        self.white_field = deepcopy(self.empty)
        self.past1_black = deepcopy(self.empty)
        self.past1_white = deepcopy(self.empty)
        self.past2_black = deepcopy(self.empty)
        self.past2_white = deepcopy(self.empty)
        self.count_turns = 0

        self.model_p = PNet()
        self.model_p = torch.nn.DataParallel(self.model_p)
        self.model_p.load_state_dict(
            torch.load("model_p{}.pth".format(number), map_location=lambda storage, loc: storage))
        self.model_p.eval()

        self.model_v = VNet()
        self.model_v = torch.nn.DataParallel(self.model_v)
        self.model_v.load_state_dict(
            torch.load("model_v{}.pth".format(number), map_location=lambda storage, loc: storage))
        self.model_v.eval()

    def get_pv(self, field):
        policy = self.model_p(field)
        policy = F.softmax(policy, dim=1)

        v = self.model_v(field)
        v = F.softmax(v, dim=1)
        v = v.detach().numpy()[0]

        return policy.detach().numpy()[0], v.argmax()

    def move(self, field, turn):
        self.past2_black = deepcopy(self.past1_black)
        self.past2_white = deepcopy(self.past1_white)
        self.past1_black = deepcopy(self.black_field)
        self.past1_white = deepcopy(self.white_field)
        self.black_field = deepcopy(field.get_black())
        self.white_field = deepcopy(field.get_white())
        turn_ = self.black_turn * turn + (not turn) * self.white_turn

        input_ = deepcopy(np.stack(
            (self.black_field, self.white_field, turn_, self.past1_black, self.past1_white, self.past2_black,
             self.past2_white)))

        policy, evaluation = self.get_pv(input_)

        possible = deepcopy(field.free)

        node = [0 for _ in range(225)]

        root = str(KEYS())

        data = {root: [policy, deepcopy(node), deepcopy(node)]}

        for _ in range(self.iterations):
            if len(possible) != 0:
                data = self.tree_search(deepcopy(data), deepcopy(possible), deepcopy(input_), deepcopy(turn))

        n_s = np.array(data[root][1])
        n_s = np.exp(n_s) / np.sum(np.exp(n_s))
        move = n_s.argmax()
        return move // 15, move % 15

    def update_field(self, field, move):
        past2_black = deepcopy(field[3])
        past2_white = deepcopy(field[4])
        past1_black = deepcopy(field[0])
        past1_white = deepcopy(field[1])
        turn = deepcopy(field[2])
        if turn:
            field[0][move // 15][move % 15] = 1.0
        else:
            field[1][move // 15][move % 15] = 1.0
        field[2] *= -1
        field[3] = past1_black
        field[4] = past1_white
        field[5] = past2_black
        field[6] = past2_white

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

            u = np.array([policy[i] / (n_s[i] + 1) for i in range(15 * 15)])

            choosing = u + q

            move = choosing.argmax()

            while move not in possible:
                choosing[move] -= self.iterations
                move = choosing.argmax()

            possible.remove(move)
            moves.add(move, black)
            made_moves.append(move)
            field = self.update_field(deepcopy(field), move)

            black = not black

            if self.check_winner(move, field.field_()):
                winner = 1
                break

        policy, evaluation = self.get_pv(field)
        node = [0 for _ in range(225)]
        data[str(moves)] = [deepcopy(policy), deepcopy(node), deepcopy(node)]

        if winner:
            evaluation = 1
        else:
            if not black and not evaluation:
                evaluation = 1

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

    def check_winner(self, move, board):
        i = move // 15
        j = move % 15

        # vertical check
        for shift in range(5):
            stones = []
            cur = 0
            for k in range(5):
                if 15 > i - k + shift >= 0 and 15 > j >= 0:
                    cur = board[i - k + shift][j]
                if cur:
                    stones.append(cur)
            if len(stones) == 5:
                return True

        # horizontal check
        for shift in range(5):
            stones = []
            cur = 0
            for k in range(5):
                if 15 > i >= 0 and 15 > j - k + shift >= 0:
                    cur = board[i][j - k + shift]
                if cur:
                    stones.append(cur)
            if len(stones) == 5:
                return True

        # diagonal check 1
        for shift in range(5):
            stones = []
            cur = 0
            for k in range(5):
                if 15 > i - k + shift >= 0 and 15 > j - k + shift >= 0:
                    cur = board[i - k + shift][j - k + shift]
                if cur:
                    stones.append(cur)
            if len(stones) == 5:
                return True

        # diagonal check 2
        for shift in range(5):
            stones = []
            cur = 0
            for k in range(5):
                if 15 > i - k + shift >= 0 and 15 > j + k - shift >= 0:
                    cur = board[i - k + shift][j + k - shift]
                if cur:
                    stones.append(cur)
            if len(stones) == 5:
                return True
        return False
