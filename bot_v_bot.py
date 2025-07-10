from dlgo.agent import naive, dlagents
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
from dlgo.encoders import get_encoder_by_name
from gui import GameGUI
import time
import argparse
import os


def list_available_models(models_dir='dl_human_games/epochs'):
    """List all available model files in the specified directory."""
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found.")
        return []
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return models


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Watch two bots play Go against each other.')
    parser.add_argument('--black', choices=['random', 'dl'], default='random', 
                        help='Black player bot type (default: random)')
    parser.add_argument('--white', choices=['random', 'dl'], default='dl', 
                        help='White player bot type (default: dl)')
    parser.add_argument('--black-model', type=str, help='Model file for black player (if using dl)')
    parser.add_argument('--white-model', type=str, help='Model file for white player (if using dl)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = list_available_models()
        if models:
            print("Available models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
        else:
            print("No models found.")
        return
    
    board_size = 19
    game = goboard.GameState.new_game(board_size)
    encoder = get_encoder_by_name('simple', board_size)
    
    # Configure black bot
    if args.black == 'random':
        bot_black = naive.RandomBot()
        black_model = None
    else:  # dl agent
        black_model = args.black_model or 'dl_human_games/epochs/small_model_epoch_5.pth'
        if not os.path.exists(black_model) and not black_model.startswith('dl_human_games/'):
            black_model = os.path.join('dl_human_games/epochs', black_model)
        bot_black = dlagents.DLAgent(encoder, black_model)
    
    # Configure white bot
    if args.white == 'random':
        bot_white = naive.RandomBot()
        white_model = None
    else:  # dl agent
        white_model = args.white_model or 'dl_human_games/epochs/small_model_epoch_5.pth'
        if not os.path.exists(white_model) and not white_model.startswith('dl_human_games/'):
            white_model = os.path.join('dl_human_games/epochs', white_model)
        bot_white = dlagents.DLAgent(encoder, white_model)
    
    bots = {
        gotypes.Player.black: bot_black,
        gotypes.Player.white: bot_white,
    }
    
    print(f"Starting game with board size {board_size}")
    print(f"Black player: {args.black}" + (f" (Model: {black_model})" if args.black == 'dl' else ""))
    print(f"White player: {args.white}" + (f" (Model: {white_model})" if args.white == 'dl' else ""))

    # Create a bot manager object that the GUI can use
    class BotManager:
        def __init__(self, bots_dict):
            self.bots = bots_dict
            
        def select_move(self, game_state):
            bot = self.bots[game_state.next_player]
            return bot.select_move(game_state)
    
    bot_manager = BotManager(bots)
    gui = GameGUI(game, bot=bot_manager, bot_players=[gotypes.Player.black, gotypes.Player.white])
    gui.run()


if __name__ == '__main__':
    main()