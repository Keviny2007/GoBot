import torch.nn as nn
import torch.nn.functional as F
from dl_human_games.layers import layers

class BetterGoCNN(nn.Module):
    def __init__(self, board_size=19):
        super(BetterGoCNN, self).__init__()
        input_shape = (11, board_size, board_size)

        self.model = layers(input_shape)

    def forward(self, x):
        return self.model(x)
