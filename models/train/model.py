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


class LSTMModel(nn.Module):
    def __init__(self, features=40, hidden=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, 50, 64)
        out = out[:, -1, :]          # (batch, 64) — last timestep only
        out = self.dropout(out)
        out = self.fc(out)            # (batch, 3)
        return out
