import pygame
import time
from Field import Field


class Visual:
    def __init__(self, field=None, sleep=0.05):
        pygame.init()
        self.size = 50  # 50 pixels per cell
        self.nodes = 16  # 15 crossing lines == draw 14 cells and two on sides
        self.display = pygame.display.set_mode((self.size * self.nodes, self.size * self.nodes))
        pygame.display.set_caption("Renju game")
        self.surface = None
        self.field = Field(start=field)
        self.sleep = sleep  # time we sleep after each move

    def get_field(self):
        return self.field

    def reset_board(self):  # resetting the field
        self.field.reset()

        self.surface = pygame.Surface((self.size * self.nodes, self.size * self.nodes))
        deck_color = (210, 180, 140)  # pretty nice color
        self.surface.fill(deck_color)  # fill the display with our nice color

        font = pygame.font.Font(None, self.size / 2)  # set some font for numbers

        # horizontal lines and numbers,,
        for i in range(self.field.get_size()):
            text = font.render(str(i + 1), 2, (0, 0, 0))
            self.surface.blit(text, (self.size + i * self.size, self.size / 3))
            pygame.draw.line(self.surface, (0, 0, 0), (self.size + i * self.size, self.size),
                             (self.size + i * self.size, self.size * (self.nodes - 1)), 2)

        # vertical lines and numbers
        for i in range(self.field.get_size()):
            text = font.render(str(i + 1), 2, (0, 0, 0))
            self.surface.blit(text, (self.size / 3, self.size - 5 + i * self.size))
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

        self.field.get_node(i, j).set_stone(stone)  # first place the stone
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
                self.highlight_winner(poses)
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
                self.highlight_winner(poses)
                return True

        # diagonal check
        for shift in range(5):
            stones = []
            poses = []
            for k in range(5):
                cur = self.field.get_node(i - k + shift, j - k + shift)
                if cur is not None:
                    stones.append(cur.get_stone())
                    poses.append([i - k + shift, j - k + shift])
            if self.check_list(stones, stone):
                self.highlight_winner(poses)
                return True

        return False

    def check_list(self, stones, stone):
        if len(stones) != 5:
            return False
        for item in stones:
            if item != stone:
                return False
        return True

    def highlight_winner(self, poses):
        color = (0, 255, 0)  # green is the victory color
        for pos in poses:
            pos_ = ((pos[0] + 1) * self.size, (pos[1] + 1) * self.size)
            pygame.draw.circle(self.surface, color, pos_, 10, 10)  # highlight the stone
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
        font = pygame.font.Font(None, self.size / 2)
        if winner is None:
            text = font.render('GAME ENDED, PRESS QUIT TO EXIT', 2, (0, 0, 0))
        else:
            text = font.render('PLAYER {} WON! PRESS QUIT TO EXIT'.format(winner), 2, (0, 0, 0))
        self.surface.blit(text, (self.size * self.nodes / 2, self.size * (self.nodes - 1) + self.size / 2))
        self.show_board()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # if you wanna quit - you're welcome
                    pygame.display.quit()
                    pygame.quit()
                    exit()
