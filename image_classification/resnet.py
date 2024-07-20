import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels