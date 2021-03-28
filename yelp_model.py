"""Defines the Yelp Model class."""

import torch
import torch.nn as nn
import torch.optim as optim

class YelpModel(nn.Module):
    """Defines the Yelp Model class."""

    def __init__(self, input_size=768 * 75, hidden_size=1024):
        super(YelpModel, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """Forward method."""
        x1 = self.relu1(self.fc1(x))
        x2 = self.relu2(self.fc2(x1))
        x3 = self.sigmoid(self.fc3(x2))
        return x3