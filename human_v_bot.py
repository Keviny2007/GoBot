from dlgo.agent import dlagents
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from gui import GameGUI
from dlgo.encoders.base import get_encoder_by_name


def main():
    board_size = 19
    game = goboard.GameState.new_game(board_size)
    encoder = get_encoder_by_name('simple', 19)
    bot = dlagents.DLAgent(encoder, 'dl_human_games/epochs/small_model_epoch_5.pth')
    gui = GameGUI(game, bot=bot, bot_players=[gotypes.Player.white])
    gui.run()


if __name__ == '__main__':
    main()