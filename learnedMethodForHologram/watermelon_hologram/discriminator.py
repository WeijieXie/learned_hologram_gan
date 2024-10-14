import torch
import torch.nn as nn


class WGANGPDiscriminator192(nn.Module):
    def __init__(
        self,
        pretrained_model_path=None,
        feature_d=32,
        cuda=True,
    ):
        super(WGANGPDiscriminator192, self).__init__()

        self.device = torch.device("cuda") if cuda else torch.device("cpu")

        self.block1 = nn.Sequential(
            nn.Conv2d(3, feature_d, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = self._make_layer(feature_d, feature_d * 2, stride=2)
        self.block3 = self._make_layer(feature_d * 2, feature_d * 4, stride=1)
        self.block4 = self._make_layer(feature_d * 4, feature_d * 8, stride=2)
        self.block5 = self._make_layer(feature_d * 8, feature_d * 16, stride=1)
        self.block6 = self._make_layer(feature_d * 16, feature_d * 32, stride=2)

        self.conv = nn.Conv2d(feature_d * 32, 1, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

        self = self.to(self.device)

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path))

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv(x)
        return self.flatten(x)


class fakeDiscriminator(nn.Module):
    def __init__(
        self,
        pretrained_model_path=None,
        feature_d=32,
        cuda=True,
    ):
        super(fakeDiscriminator, self).__init__()
        self.a = nn.parameter.Parameter(torch.tensor([1.0]))
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self._requires_grad = False

    def forward(self, _):
        return torch.tensor([0.0], device=self.device)
