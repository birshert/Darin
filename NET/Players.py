import numpy as np
import pygame
from Net import *
import torch
import torch.nn
import torch.nn.functional as F
from copy import deepcopy


class RandomPlayer:
    def __init__(self):
        self.possible_ = []

    def possible_moves(self, field):
        moves = []
        for i in range(field.get_size()):
            for j in range(field.get_size()):
                if field.get_node(i, j).is_empty():
                    moves.append([i, j])
        self.possible_ = moves

    def move_(self, field, turn):
        self.possible_moves(field)
        pos = np.random.randint(0, len(self.possible_))
        return self.possible_[pos]


class HumanPlayer:
    def __init__(self):
        pass

    def move_(self, field, turn):
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
    def __init__(self, number):
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

        self.empty = np.array([[0.0 for _ in range(15)] for _ in range(15)])
        self.white_turn = np.array([[-1.0 for _ in range(15)] for _ in range(15)])
        self.black_turn = np.array([[+1.0 for _ in range(15)] for _ in range(15)])
        self.past1_black = deepcopy(self.empty)
        self.past1_white = deepcopy(self.empty)
        self.past2_black = deepcopy(self.empty)
        self.past2_white = deepcopy(self.empty)
        self.count_turns = 0

    def move_(self, field, turn):
        black_field = deepcopy(field.get_black())
        white_field = deepcopy(field.get_white())
        turn_ = self.black_turn * turn + (not turn) * self.white_turn
        input_ = deepcopy(np.stack(
            (black_field, white_field, turn_, self.past1_black, self.past1_white, self.past2_black, self.past2_white),
            axis=0))
        input_ = torch.stack([torch.from_numpy(input_).type(torch.FloatTensor)])
        policy = self.model_p(input_)
        v = self.model_v(input_)
        v = F.softmax(v, dim=1)
        output_ = F.softmax(policy, dim=1)
        prob = output_.detach().numpy()[0]
        print(max(prob))
        while True:
            move = np.random.choice(np.arange(0, 225), p=prob)
            if field.get_node(move // 15, move % 15).is_empty():
                if self.count_turns > 1:
                    self.past1_black = deepcopy(black_field)
                    self.past1_white = deepcopy(white_field)
                if self.count_turns > 2:
                    self.past2_black = deepcopy(self.past1_black)
                    self.past2_white = deepcopy(self.past1_white)
                    self.past1_black = deepcopy(black_field)
                    self.past1_white = deepcopy(white_field)
                self.count_turns += 1
                return move // 15, move % 15
            prob[move] = 0
            prob = self.softmax(prob)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
