import numpy as np
from copy import deepcopy

import pygame
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Visual:
    def __init__(self, field=None, sleep=0):
        pygame.init()
        self.size = 50  # 50 pixels per cell
        self.nodes = 16  # 15 crossing lines == draw 14 cells and two on sides
        self.display = pygame.display.set_mode((self.size * self.nodes, self.size * self.nodes))
        pygame.display.set_caption("Renju game")
        self.surface = pygame.Surface((self.size * self.nodes, self.size * self.nodes))
        self.start_color = (135, 206, 235)  # dark golden rod for the victory
        self.surface.fill(self.start_color)
        self.font = pygame.font.Font(None, self.size // 2)  # set some font for numbers etc.
        hello = self.font.render('PRESS B FOR PLAYING FOR BLACK, W FOR WHITE', True, (0, 0, 0))
        self.surface.blit(hello, (200, 350))
        self.field = Field(start=field)
        self.sleep = sleep  # time we sleep after each move
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()

    def get_field(self):
        return self.field

    def reset_board(self):  # resetting the field
        self.field.reset()

        deck_color = (210, 180, 140)  # pretty nice color
        self.surface.fill(deck_color)  # fill the display with our nice color

        change = {1: 'a',
                  2: 'b',
                  3: 'c',
                  4: 'd',
                  5: 'e',
                  6: 'f',
                  7: 'g',
                  8: 'h',
                  9: 'j',
                  10: 'k',
                  11: 'l',
                  12: 'm',
                  13: 'n',
                  14: 'o',
                  15: 'p'}

        # horizontal lines and numbers
        for i in range(self.field.get_size()):
            text = self.font.render(change[i + 1], 2, (0, 0, 0))
            self.surface.blit(text, (self.size + i * self.size, self.size // 3))
            pygame.draw.line(self.surface, (0, 0, 0), (self.size + i * self.size, self.size),
                             (self.size + i * self.size, self.size * (self.nodes - 1)), 2)

        # vertical lines and numbers
        for i in range(self.field.get_size()):
            text = self.font.render(str(i + 1), 2, (0, 0, 0))
            self.surface.blit(text, (self.size // 3, self.size - 5 + i * self.size))
            pygame.draw.line(self.surface, (0, 0, 0), (self.size, self.size + i * self.size),
                             (self.size * (self.nodes - 1), self.size + i * self.size), 2)

        self.show_board()  # show what we have at the beginning

    def show_board(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # if you wanna quit - you're welcome
                pygame.display.quit()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:  # if you wanna pause - you're welcome
                if event.key == pygame.K_SPACE:
                    self.pause()

        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()
        time.sleep(self.sleep)

    def draw_field(self):
        for i in range(1, 16):
            for j in range(1, 16):
                if not self.field.get_node(i, j).is_empty():  # if there is a stone
                    color = self.field.get_node(i, j).color()
                    pygame.draw.circle(self.surface, color, ((i + 1) * self.size, (j + 1) * self.size), 10, 10)
        self.show_board()

    def move(self, move, is_black):
        i, j = move

        if is_black:
            stone = 1  # actually black
        else:
            stone = -1  # guess what (etihw)

        self.field.make_move(i, j, stone)  # first place the stone
        color = self.field.get_node(i, j).color()  # get it color
        pygame.draw.circle(self.surface, color, ((i + 1) * self.size, (j + 1) * self.size), 10, 10)  # draw the stone
        self.show_board()  # show the board and we're great

        return self.check_winner(move), self.field

    def check_winner(self, move):
        i, j = move
        stone = self.field.get_node(i, j).get_stone()

        # vertical check
        for shift in range(5):
            stones = []
            poses = []
            for k in range(5):
                cur = self.field.get_node(i - k + shift, j)
                if cur is not None:
                    stones.append(cur.get_stone())
                    poses.append([i - k + shift, j])
            if self.check_list(stones, stone):
                self.highlight_winner(poses, stone)
                return True

        # horizontal check
        for shift in range(5):
            stones = []
            poses = []
            for k in range(5):
                cur = self.field.get_node(i, j - k + shift)
                if cur is not None:
                    stones.append(cur.get_stone())
                    poses.append([i, j - k + shift])
            if self.check_list(stones, stone):
                self.highlight_winner(poses, stone)
                return True

        # diagonal check 1
        for shift in range(5):
            stones = []
            poses = []
            for k in range(5):
                cur = self.field.get_node(i - k + shift, j - k + shift)
                if cur is not None:
                    stones.append(cur.get_stone())
                    poses.append([i - k + shift, j - k + shift])
            if self.check_list(stones, stone):
                self.highlight_winner(poses, stone)
                return True

        # diagonal check 2
        for shift in range(5):
            stones = []
            poses = []
            for k in range(5):
                cur = self.field.get_node(i - k + shift, j + k - shift)
                if cur is not None:
                    stones.append(cur.get_stone())
                    poses.append([i - k + shift, j + k - shift])
            if self.check_list(stones, stone):
                self.highlight_winner(poses, stone)
                return True

        return False

    def check_list(self, stones, stone):
        if len(stones) != 5:
            return False
        for item in stones:
            if item != stone:
                return False
        return True

    def highlight_winner(self, poses, stone):
        color = (stone == -1) * (255, 255, 255) + (stone == 1) * (0, 0, 0)
        for pos in poses:
            pos_ = ((pos[0] + 1) * self.size, (pos[1] + 1) * self.size)
            pygame.draw.circle(self.surface, color, pos_, self.size // 10 * 3,
                               self.size // 10 * 3)  # highlight the stone
        self.show_board()

    def pause(self):
        paus = True
        while paus:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # touch space again to play
                        paus = False
                if event.type == pygame.QUIT:  # if you wanna quit during pause - you're welcome
                    pygame.display.quit()
                    pygame.quit()
                    exit()

    def end(self, winner=None):
        if winner is None:
            text = self.font.render('GAME ENDED WITH DRAW, PRESS QUIT TO EXIT', 2, (0, 0, 0))
        else:
            text = self.font.render('{} PLAYER WON! PRESS QUIT TO EXIT, R FOR RESTART'.format(winner), 2, (0, 0, 0))
        self.surface.blit(text, (self.size, self.size * (self.nodes - 1) + self.size // 2))
        self.show_board()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # if you wanna quit - you're welcome
                    pygame.display.quit()
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        running = False


class HumanPlayer:
    def __init__(self):
        pass

    @staticmethod
    def move_(field, turn):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pos = [event.pos[0], event.pos[1]]
                        pos[0] -= 25
                        pos[1] -= 25
                        pos[0] //= 50
                        pos[1] //= 50
                        if field.get_node(pos[0], pos[1]).is_empty():
                            return pos
                        else:
                            print("INVALID MOVE\nCHOOSE ANOTHER\n")
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    exit()


class AI:
    def __init__(self):
        self.mcts = MCTS(5)

    def move_(self, field, turn):
        ret, move = self.trick(deepcopy(field.get_black() * turn + field.get_white() * (not turn)), field.free)
        if ret:
            return move // 15, move % 15
        ret, move = self.trick(deepcopy(field.get_black() * (not turn) + field.get_white() * turn), field.free)
        if ret:
            return move // 15, move % 15

        return self.mcts.move(field, turn)

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


class MCTS:
    def __init__(self, time):
        self.iterations_time = time
        self.t = 1
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
            torch.load("model_p.pth", map_location=lambda storage, loc: storage))
        self.model.eval()

        self.model2 = VNet()
        self.model2.load_state_dict(
            torch.load("model_v.pth", map_location=lambda storage, loc: storage))
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
        self.black = turn
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

        move = n_s.argmax()

        self.count_turns += 2
        if self.count_turns > 30:
            self.t = 0.5
        if self.count_turns > 50:
            self.t = 0.1

        return move // 15, move % 15

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
                choosing[move] -= 100
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

        if black:
            evaluation = evaluation[1] - evaluation[0]
        else:
            evaluation = evaluation[0] - evaluation[1]

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


class Game:
    def __init__(self):
        self.vis = Visual()
        self.field = self.vis.get_field()
        self.turn = True
        self.curr_player = None
        self.running = False
        self.count_turns = 0

    def game_(self, player1=None, player2=None):
        self.vis.reset_board()
        if player1 is None:
            player1 = HumanPlayer()
        if player2 is None:
            player2 = AI()
        self.curr_player = player1
        self.turn = True
        for _ in range(15 * 15):
            self.count_turns += 1
            position = self.curr_player.move_(self.field, self.turn)
            winner, self.field = self.vis.move(position, self.turn)
            if winner:
                if self.curr_player == player1:
                    self.vis.end('BLACK')
                    self.running = False
                    break
                else:
                    self.vis.end('WHITE')
                    self.running = False
                    break
            self.turn = not self.turn
            if self.curr_player == player1:
                self.curr_player = player2
            else:
                self.curr_player = player1
        if self.count_turns == 225:
            self.vis.end()

    def play_game(self, player1=None, player2=None, id_=None):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # if you wanna quit - you're welcome
                    pygame.display.quit()
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        player1 = HumanPlayer()
                        player2 = AI()
                        self.game_(player1, player2)
                        break
                    elif event.key == pygame.K_w:
                        player1 = HumanPlayer()
                        player2 = AI()
                        self.game_(player2, player1)
                        break


def main():
    while True:
        a = Game()
        a.play_game()


if __name__ == "__main__":
    main()
