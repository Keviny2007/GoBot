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
    bot_white = dlagents.DLAgent(encoder, 'dl_human_games/epochs/small_model_epoch_5.pth')

    bots = {
        gotypes.Player.black: bot_black,
        gotypes.Player.white: bot_white,
    }

    gui = GameGUI(game, bot=bots, bot_players=[gotypes.Player.black, gotypes.Player.white])
    gui.run()


if __name__ == '__main__':
    main()