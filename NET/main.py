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
            move = self.curr_player.move_(poss)
            self.vis.move(move, self.turn)
            self.turn = not self.turn
            self.curr_player = (self.curr_player == player1) * player2 + (self.curr_player == player2) * player1


a = Game()

a.start_game()