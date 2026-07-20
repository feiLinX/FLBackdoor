import torch
import torch.nn as nn


class FangCNN(nn.Module):
    def __init__(self):
        super(FangCNN, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(3, 30, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(30, 50, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2),
                                nn.Flatten(),
                                nn.Linear(1250, 512),
                                nn.ReLU(),
                                nn.Linear(512, 10)
        )

    def init_ones(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_zeros(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


    def init_xavier(self,m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight, gain=2.24)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        return x