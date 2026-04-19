import torch
import torch.nn as nn

class HeartModel(nn.Module):
    def __init__(self):
        super(HeartModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(13, 16),   # ✅ FIXED INPUT SIZE
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)