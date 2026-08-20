import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, y):
        t = torch.cat([x, y], dim=1)
        return self.net(t)

    def mi_loss(self, x, y):
        joint = self.forward(x, y).mean()
        y_shuffle = y[torch.randperm(y.size(0))]
        marg = torch.log(torch.exp(self.forward(x, y_shuffle)).mean() + 1e-8)
        mi = joint - marg
        return -mi
