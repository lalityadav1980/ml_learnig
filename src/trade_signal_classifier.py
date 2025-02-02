#trade_signal_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradeSignalClassifier(nn.Module):
    """
    A simple feed-forward neural network for trade signal classification.
    Input: state features (e.g., ADX, TR, ATR, bands, numeric STX, reward, etc.)
    Output: logits for 3 classes (0: NoTrade, 1: Buy, 2: Sell)
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(TradeSignalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        return logits


class StopLossRegressor(nn.Module):
    """
    A simple feed-forward network for stop-loss regression.
    Input: state features (can be the same as the classifier or extended)
    Output: a single continuous value representing the recommended stop-loss level.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super(StopLossRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)
        return out
