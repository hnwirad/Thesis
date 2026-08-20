import argparse
import torch
from torch import optim
from models.mine import MINE
from losses import PerceptualLoss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mine = MINE(x_dim=512, y_dim=64).to(device)
    mine_optim = optim.Adam(mine.parameters(), lr=1e-4)
    perc = PerceptualLoss()

    for epoch in range(args.epochs):
        for batch in []:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    train(args)
