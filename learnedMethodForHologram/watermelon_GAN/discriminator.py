import torch
import torch.nn as nn


class WGANGPDiscriminator192(nn.Module):
    def __init__(
        self,
        pretrained_model_path=None,
        feature_d=64,
        cuda=True,
    ):
        super(WGANGPDiscriminator192, self).__init__()
        self.device = torch.device("cuda") if cuda else torch.device("cpu")

        self.part1 = nn.Sequential(
            # Input: N x img_channels x 192 x 192
            nn.LazyConv2d(feature_d, kernel_size=4, stride=2, padding=1),  # 96x96
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(feature_d * 2, kernel_size=4, stride=2, padding=1),  # 48x48
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(feature_d * 4, kernel_size=4, stride=2, padding=1),  # 24x24
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(feature_d * 8, kernel_size=4, stride=2, padding=1),  # 12x12
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(feature_d * 16, kernel_size=4, stride=2, padding=1),  # 6x6
            nn.BatchNorm2d(feature_d * 16),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(1, kernel_size=6, stride=1, padding=0),  # 1x1
        ).to(self.device)

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path))

    def forward(self, x):
        x = self.part1(x)
        return x.view(-1)
