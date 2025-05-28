from dlgo.agent import naive
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
from gui import GameGUI
import time

def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)

    bot = naive.RandomBot()
    gui = GameGUI(game, bot=bot, bot_players=[gotypes.Player.black, gotypes.Player.white])
    gui.run()
    # bots = {
    #     gotypes.Player.black: naive.RandomBot(),
    #     gotypes.Player.white: naive.RandomBot(),
    # }
    # while not game.is_over():
    #     time.sleep(0.3)

    #     print(chr(27) + "[2J")
    #     print_board(game.board)
    #     bot_move = bots[game.next_player].select_move(game)
    #     print_move(game.next_player, bot_move)
    #     game = game.apply_move(bot_move)


if __name__ == '__main__':
    main()