import torch.nn as nn
import torch


class Baseline(nn.Module):
    def __init__(self, window=50, features=40):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.l1 = nn.Linear(window * features, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1)

        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.l3(x)

        return x
