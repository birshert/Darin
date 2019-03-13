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
        self.model = Net()
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load("model{}.pth".format(number), map_location=lambda storage, loc: storage))
        self.model.eval()

    def move_(self, field, turn):
        field_ = deepcopy(field.field_())
        print(field_)
        input_ = deepcopy(torch.stack([torch.from_numpy(field_).type(torch.FloatTensor)]))
        output = F.softmax(self.model(input_), dim=1)
        while True:
            move = output.data.max(1, keepdim=True)[1]
            print(move)
            if field.get_node(move // 15, move % 15).is_empty():
                return move // 15, move % 15
            output[0][move] -= 1
