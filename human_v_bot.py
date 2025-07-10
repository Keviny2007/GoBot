from dlgo.agent import dlagents
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from gui import GameGUI
from dlgo.encoders.base import get_encoder_by_name
import os
import argparse


def list_available_models(models_dir='dl_human_games/epochs'):
    """List all available model files in the specified directory."""
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found.")
        return []
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return models


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Play Go against a bot.')
    parser.add_argument('--board-size', type=int, default=19, help='Board size (default: 19)')
    parser.add_argument('--bot-color', choices=['black', 'white'], default='white', 
                        help='Bot player color (default: white)')
    parser.add_argument('--model', type=str, help='Specific model file to use')
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
    
    # If no model specified, let user choose one
    model_path = args.model
    if not model_path:
        models = list_available_models()
        if not models:
            print("No models found. Please train a model first.")
            return
            
        print("Available models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
            
        while True:
            try:
                choice = int(input("Select a model (enter number): "))
                if 1 <= choice <= len(models):
                    model_path = os.path.join('dl_human_games/epochs', models[choice-1])
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        # If model was specified but without full path, assume it's in the default directory
        if not os.path.exists(model_path):
            model_path = os.path.join('dl_human_games/epochs', model_path)
    
    # Set up the game
    board_size = args.board_size
    game = goboard.GameState.new_game(board_size)
    encoder = get_encoder_by_name('simple', board_size)
    
    bot = dlagents.DLAgent(encoder, model_path)
    
    # Determine bot color
    bot_color = gotypes.Player.white if args.bot_color == 'white' else gotypes.Player.black
    gui = GameGUI(game, bot=bot, bot_players=[bot_color])
    
    print(f"Starting game with board size {board_size}")
    print(f"Bot is playing as {args.bot_color}")
    print(f"Using model: {model_path}")
    
    gui.run()


if __name__ == '__main__':
    main()