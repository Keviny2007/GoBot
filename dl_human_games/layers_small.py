import torch
import torch.nn as nn

def layers(input_shape):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.ConstantPad2d(3, 0),  # Equivalent to ZeroPadding2D(padding=3)
                nn.Conv2d(input_shape[0], 48, kernel_size=7),
                nn.ReLU(),

                nn.ConstantPad2d(2, 0),
                nn.Conv2d(48, 32, kernel_size=5),
                nn.ReLU(),

                nn.ConstantPad2d(2, 0),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.ReLU(),

                nn.ConstantPad2d(2, 0),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.ReLU(),
            )

            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                dummy_out = self.features(dummy)
                self.flat_size = dummy_out.view(1, -1).shape[1]

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flat_size, 512),
                nn.ReLU(),
                nn.Linear(512, 361)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return Model()
