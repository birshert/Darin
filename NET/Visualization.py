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
        self.main_surface = None
        self.field = Field(start=field)
        self.sleep = sleep  # time we sleep after each move

    def reset_board(self):  # resetting the field
        self.field.reset()

        self.main_surface = pygame.Surface((self.size * self.nodes, self.size * self.nodes))
        deck_color = (210, 180, 140)  # pretty nice color
        self.main_surface.fill(deck_color)  # fill the display with our nice color

        font = pygame.font.Font(None, self.size / 2)  # set some font for numbers

        # horizontal lines and numbers
        for i in range(self.field.get_size()):
            text = font.render(str(i + 1), 2, (0, 0, 0))
            self.main_surface.blit(text, (self.size + i * self.size, self.size / 3))
            pygame.draw.line(self.main_surface, (0, 0, 0), (self.size + i * self.size, self.size),
                             (self.size + i * self.size, self.size * (self.nodes - 1)), 2)

        # vertical lines and numbers
        for i in range(self.field.get_size()):
            text = font.render(str(i + 1), 2, (0, 0, 0))
            self.main_surface.blit(text, (self.size / 3, self.size - 5 + i * self.size))
            pygame.draw.line(self.main_surface, (0, 0, 0), (self.size, self.size + i * self.size),
                             (self.size * (self.nodes - 1), self.size + i * self.size), 2)

        self.show_board()  # show what we have at the beginning

    def show_board(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # if you wanna quit - you're welcome
                pygame.display.quit()
                pygame.quit()
                exit()

        self.display.blit(self.main_surface, (0, 0))
        pygame.display.flip()
        time.sleep(self.sleep)

    def draw_field(self):
        for i in range(1, 16):
            for j in range(1, 16):
                if not self.field.get_node(i, j).is_empty():  # if there is a stone
                    color = self.field.get_node(i, j).color()
                    pygame.draw.circle(self.main_surface, color, (i * self.size, j * self.size), 10, 10)
        self.show_board()

    def move(self, move, is_black):
        i, j = move

        if is_black:
            stone = 1  # actually black
        else:
            stone = -1  # guess what (etihw)

        self.field.get_node(i, j).set_stone(stone)
        color = self.field.get_node(i, j).color()
        pygame.draw.circle(self.main_surface, color, (i * self.size, j * self.size), 10, 10)
        self.show_board()
