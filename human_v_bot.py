from dlgo.agent import bad
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from gui import GameGUI
from dlgo.encoders.base import get_encoder_by_name

# from six.moves import input

def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    encoder = get_encoder_by_name('oneplane', 9)
    bot = bad.BadAgent(encoder)
    gui = GameGUI(game, bot=bot, bot_players=[gotypes.Player.white])
    gui.run()
    # while not game.is_over():
    #     print(chr(27) + "[2J")
    #     print_board(game.board)
    #     if game.next_player == gotypes.Player.black:
    #         human_move = input('-- ')
    #         point = point_from_coords(human_move.strip())
    #         move = goboard.Move.play(point)
    #     else:
    #         move = bot.select_move(game)
    #     print_move(game.next_player, move)
    #     game = game.apply_move(move)

if __name__ == '__main__':
    main()