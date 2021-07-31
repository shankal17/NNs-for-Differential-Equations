import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(1, 32)
        self.fc_2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.fc_1(x))
        out = self.sigmoid(self.fc_2(out))

        return out

if __name__ == '__main__':
    nn = MLP()
    x = torch.Tensor([1.0])
    print(nn(x))