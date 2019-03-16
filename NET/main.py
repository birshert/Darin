from Visualization import *
from Players import *


class Game:
    def __init__(self):
        self.vis = Visual()
        self.field = self.vis.get_field()
        self.turn = True
        self.curr_player = None
        self.running = False

    def game_(self, player1=None, player2=None, id_=None):
        self.vis.reset_board()
        if player1 is None:
            player1 = AI(id_)
        if player2 is None:
            player2 = HumanPlayer()
        self.curr_player = player1
        self.turn = True
        for num in range(15 * 15):
            position = self.curr_player.move_(self.field, self.turn)
            winner, self.field = self.vis.move(position, self.turn)
            if winner:
                if self.curr_player == player1:
                    self.vis.end('BLACK')
                else:
                    self.vis.end('WHITE')
            self.turn = not self.turn
            if self.curr_player == player1:
                self.curr_player = player2
            else:
                self.curr_player = player1
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
                    if event.key == pygame.K_g:
                        self.game_(player1, player2, id_)
                        self.running = False


a = Game()

a.play_game(id_=3)