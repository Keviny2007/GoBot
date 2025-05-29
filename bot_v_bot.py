from dlgo.agent import naive, dlagents
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
from dlgo.encoders import get_encoder_by_name
from gui import GameGUI
import time

def main():
    board_size = 19
    game = goboard.GameState.new_game(board_size)

    encoder = get_encoder_by_name('simple', 19)
    bot_black = naive.RandomBot()
    bot_white = dlagents.DLAgent(encoder, 'dl_human_games/small_model_epoch_5.pth')

    bots = {
        gotypes.Player.black: bot_black,
        gotypes.Player.white: bot_white,
    }

    gui = GameGUI(game, bot=bots, bot_players=[gotypes.Player.black, gotypes.Player.white])
    gui.run()
    
    # while not game.is_over():
    #     time.sleep(0.3)

    #     print(chr(27) + "[2J")
    #     print_board(game.board)
    #     bot_move = bots[game.next_player].select_move(game)
    #     print_move(game.next_player, bot_move)
    #     game = game.apply_move(bot_move)


if __name__ == '__main__':
    main()