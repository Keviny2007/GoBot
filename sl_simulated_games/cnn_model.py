# cnn_model.py
import torch.nn as nn
import torch.nn.functional as F

class GoCNN(nn.Module):
    def __init__(self, board_size=9):
        super(GoCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(48 * (board_size // 2)**2, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
