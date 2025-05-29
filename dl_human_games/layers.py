import torch
import torch.nn as nn

def layers(input_shape):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.ZeroPad2d(3),
                nn.Conv2d(input_shape[0], 64, kernel_size=7),
                nn.ReLU(),

                nn.ZeroPad2d(2),
                nn.Conv2d(64, 64, kernel_size=5),
                nn.ReLU(),

                nn.ZeroPad2d(2),
                nn.Conv2d(64, 64, kernel_size=5),
                nn.ReLU(),

                nn.ZeroPad2d(2),
                nn.Conv2d(64, 48, kernel_size=5),
                nn.ReLU(),

                nn.ZeroPad2d(2),
                nn.Conv2d(48, 48, kernel_size=5),
                nn.ReLU(),

                nn.ZeroPad2d(2),
                nn.Conv2d(48, 32, kernel_size=5),
                nn.ReLU(),

                nn.ZeroPad2d(2),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.ReLU()
            )
            # Compute flattened size
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_shape)
                out = self.features(dummy_input)
                self.flattened_size = out.view(1, -1).size(1)

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flattened_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 361)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return Model()
