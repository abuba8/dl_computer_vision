import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, in_channels, num_classes, config):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = 3

        layers = []
        input_channels = self.in_channels

        for f in config:
            if f == 'M':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            else:
                layers.append(nn.Conv2d(input_channels, f, kernel_size=self.kernel_size, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
                input_channels = f

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Configuration for VGG-16
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# Example usage:
# model = VGG(in_channels=3, num_classes=1000, config=vgg16_config)
