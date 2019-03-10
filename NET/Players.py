import numpy as np
import pygame
from Net import *
import torch


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
                        pos[0] /= 50
                        pos[1] /= 50
                        if field.get_node(pos[0], pos[1]).is_empty():
                            return pos
                        else:
                            print("INVALID MOVE\nCHOOSE ANOTHER\n")
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    exit()


class AI:
    def __init__(self, path):
        self.model = Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.white_ = np.array([[-1 for _ in range(15)] for _ in range(15)])
        self.black_ = np.array([[1 for _ in range(15)] for _ in range(15)])
        self.empty = np.array([0 for _ in range(15 * 15)])

    def move_(self, field, turn):
        if turn:
            turn_ = self.black_
        else:
            turn_ = self.white_

        black_field = np.array([[0 for _ in range(15)] for _ in range(15)])
        white_field = np.array([[0 for _ in range(15)] for _ in range(15)])
        for i in range(15):
            for j in range(15):
                stone = field.get_node(i, j).get_stone()
                if stone == 1:
                    black_field[i][j] = stone
                elif stone == -1:
                    white_field[i][j] = stone
        x = []
        x.append(np.stack((black_field, white_field, turn_), axis=-1))
        output1, _ = self.model(torch.tensor(x).type(torch.FloatTensor))
        while True:
            move = output1.data.max(1, keepdim=True)[1].item()
            if field.get_node(move // 15, move % 15).is_empty():
                return [move // 15, move % 15]
            output1[0][move] -= 5
