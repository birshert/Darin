from Visualization import *
from Players import *


class Game:
    def __init__(self):
        self.vis = Visual()
        self.vis.reset_board()
        self.field = self.vis.get_field()
        self.turn = True
        self.curr_player = None

    def start_game(self, player1=None, player2=None):
        if player1 is None:
            player1 = HumanPlayer()
        if player2 is None:
            player2 = RandomPlayer()
        self.curr_player = player1
        for num in range(15 * 15):
            position = self.curr_player.move_(self.field, num)
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


a = Game()

a.start_game()
