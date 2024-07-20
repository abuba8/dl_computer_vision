import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_1 = nn.Sequential(

            nn.Conv2d(in_channels=self.in_channels, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=self.out_channels)
        )

    def forward(self, x):
        x = self.stack_1(x)
        x = self.stack_2(x)
        x = self.classifier(x)
        return x