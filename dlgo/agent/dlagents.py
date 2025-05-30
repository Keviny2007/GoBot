import torch
import numpy as np
from dlgo.agent.base import Agent
from dlgo.goboard import Move
from dlgo.encoders.base import get_encoder_by_name
from dl_human_games.better_cnn_model import BetterGoCNN
import os

class DLAgent(Agent):
    def __init__(self, encoder, model_path):
        super().__init__()
        self.encoder = encoder
        self.model = BetterGoCNN()
        print("Current working directory:", os.getcwd())

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze(0).numpy()

    def select_move(self, game_state):
        move_probs = self.predict(game_state)

        move_probs = move_probs ** 3
        move_probs = np.clip(move_probs, 1e-6, 1 - 1e-6)
        move_probs = move_probs / np.sum(move_probs)

        num_moves = self.encoder.board_width * self.encoder.board_height
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(Move.play(point)):
                return Move.play(point)

        return Move.pass_turn()
