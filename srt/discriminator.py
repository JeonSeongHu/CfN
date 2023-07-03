import torch
import torch.nn as nn
import torchvision.models as models

class DiscriminatorResNet18(nn.Module):
    def __init__(self):
        super(DiscriminatorResNet18, self).__init__()

        # Load the ResNet18 model
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features

        # Replace the last fully connected layer for binary classification
        self.resnet.fc = nn.Linear(num_ftrs, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x.squeeze()

class DiscriminatorResNet50(nn.Module):
    def __init__(self):
        super(DiscriminatorResNet50, self).__init__()

        # Load the ResNet18 model
        self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features

        # Replace the last fully connected layer for binary classification
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x.squeeze()



class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


# Discriminator은 patch gan을 사용합니다.
# Patch Gan: 이미지를 16x16의 패치로 분할하여 각 패치가 진짜인지 가짜인지 식별합니다.
# low-frequency에서 정확도가 향상됩니다.

class ResNetDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(ResNetDiscriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(1, 3, 100, 100)
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)