from Visualization import Visual
from Functions import possible_moves
from Players import RandomPlayer


class Game:
    def __init__(self):
        self.vis = Visual()
        self.vis.reset_board()
        self.field = self.vis.get_field()
        self.turn = True
        self.curr_player = None

    def start_game(self, player1=None, player2=None):
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        self.curr_player = player1
        for _ in range(100):
            poss = possible_moves(self.field)
            position = self.curr_player.move_(poss)
            self.vis.move(position, self.turn)
            self.turn = not self.turn
            if self.curr_player == player1:
                self.curr_player = player2
            else:
                self.curr_player = player1


a = Game()

a.start_game()
