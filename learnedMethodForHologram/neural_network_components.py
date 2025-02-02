import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.convolution_layer_1 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=(1, 1), stride=strides
        )
        self.convolution_layer_2 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=(1, 1)
        )

        if use_1x1conv:
            self.convolution_layer_3 = nn.LazyConv2d(
                num_channels, kernel_size=1, stride=strides
            )
        else:
            self.convolution_layer_3 = None

        self.batch_norm_layer_1 = nn.LazyBatchNorm2d()
        self.batch_norm_layer_2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.batch_norm_layer_1(self.convolution_layer_1(X)))
        Y = self.batch_norm_layer_2(self.convolution_layer_2(Y))
        if self.convolution_layer_3:
            X = self.convolution_layer_3(X)
        Y += X
        return F.relu(Y)


class SymmetricConv2d(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(SymmetricConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.center = kernel_size // 2

        # Create a single parameter for each unique distance
        unique_distances = self.get_unique_distances()
        self.params = nn.Parameter(torch.abs(torch.randn(len(unique_distances))))
        self.bias = nn.Parameter(torch.zeros(1))
        self.distance_map = self.create_distance_map(unique_distances)

    def get_unique_distances(self):
        center = self.kernel_size // 2
        distances = set()
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dist = (i - center) ** 2 + (j - center) ** 2
                distances.add(dist)
        return sorted(distances)

    def create_distance_map(self, unique_distances):
        center = self.kernel_size // 2
        distance_map = torch.zeros(
            (self.kernel_size, self.kernel_size), dtype=torch.long
        )
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dist = (i - center) ** 2 + (j - center) ** 2
                distance_map[i, j] = unique_distances.index(dist)
        return distance_map

    def forward(self, x):
        # Create weight matrix with shared parameters
        weight = self.params[self.distance_map]
        weight = weight.unsqueeze(0).unsqueeze(0)

        # Convolve input with symmetric kernel
        out = F.conv2d(x, weight, self.bias, padding=self.padding)
        return out


class ChannelWiseSymmetricConv(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(ChannelWiseSymmetricConv, self).__init__()
        self.conv_r = SymmetricConv2d(kernel_size, padding)
        self.conv_g = SymmetricConv2d(kernel_size, padding)
        self.conv_b = SymmetricConv2d(kernel_size, padding)

    def forward(self, x):
        x_r = x[:, 0:1, :, :]
        x_g = x[:, 1:2, :, :]
        x_b = x[:, 2:3, :, :]

        out_r = self.conv_r(x_r)
        out_g = self.conv_g(x_g)
        out_b = self.conv_b(x_b)

        out = torch.cat((out_r, out_g, out_b), dim=1)
        return out


class fakeChannelWiseSymmetricConv(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(fakeChannelWiseSymmetricConv, self).__init__()

    def forward(self, x):
        return x


class miniResNet(nn.Module):
    def __init__(self, output_channels=3):
        super(miniResNet, self).__init__()
        self.output_channels = output_channels
        self.net = nn.Sequential(
            self.part_1(),  # conv
            self.part_2(),  # residual blocks
            self.part_3(),  # global pooling and dense layer
        )

    def part_1(self):
        return nn.Sequential(
            nn.LazyConv2d(32, kernel_size=7, stride=1, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # remove pooling layer to keep the same shape
        )

    def part_2(self):
        return nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(64, use_1x1conv=True, strides=1),
            ResidualBlock(64),
        )

    def part_3(self):
        return nn.Sequential(
            nn.LazyConv2d(out_channels=self.output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.net(X)


class ResNet(nn.Module):
    def __init__(self, output_channels=3):
        super(ResNet, self).__init__()
        self.output_channels = output_channels
        self.net = nn.Sequential(
            self.part_1(),  # conv
            self.part_2(),  # residual blocks
            self.part_3(),  # global pooling and dense layer
        )

    def part_1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=1, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # remove pooling layer to keep the same shape
        )

    def part_2(self):
        return nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(128, use_1x1conv=True, strides=1),
            ResidualBlock(128),
            ResidualBlock(256, use_1x1conv=True, strides=1),
            ResidualBlock(256),
            ResidualBlock(512, use_1x1conv=True, strides=1),
            ResidualBlock(512),
        )

    def part_3(self):
        return nn.Sequential(
            nn.LazyConv2d(out_channels=self.output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.net(X)


class ResNetPOH(ResNet):
    def __init__(self, output_channels=3):
        super(ResNetPOH, self).__init__(output_channels)

    def forward(self, X):
        return 2 * torch.pi * super(ResNetPOH, self).forward(X)


class miniUNet(nn.Module):
    def __init__(self, output_channels=1):
        super(miniUNet, self).__init__()

        self.output_channels = output_channels
        # Encoder
        self.encoder1 = nn.Sequential(
            self.conv_block(16),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(32),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(64),
            nn.LazyConvTranspose2d(32, kernel_size=2, stride=2),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            self.conv_block(32),
            nn.LazyConvTranspose2d(16, kernel_size=2, stride=2),
        )

        self.decoder2 = self.conv_block(16)

        # Final layer
        self.final_layer = nn.Sequential(
            nn.LazyConv2d(self.output_channels, kernel_size=1),
            nn.Sigmoid(),  # to ensure the output is in the range of [0, 1]
        )

    def conv_block(self, out_channels):
        return nn.Sequential(
            ResidualBlock(out_channels, use_1x1conv=True),
        )

    def forward(self, X):
        encoder1 = self.encoder1(X)
        encoder2 = self.encoder2(encoder1)

        bottleneck = self.bottleneck(encoder2)

        decoder1 = self.decoder1(torch.cat((encoder2, bottleneck), dim=1))
        decoder2 = self.decoder2(torch.cat((encoder1, decoder1), dim=1))

        return self.final_layer(decoder2)


class UNet(nn.Module):
    def __init__(self, output_channels=6):
        super(UNet, self).__init__()

        self.output_channels = output_channels
        # Encoder
        self.encoder1 = nn.Sequential(
            self.conv_block(64),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(128),
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(256),
        )

        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(512),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(1024),
            nn.LazyConvTranspose2d(512, kernel_size=2, stride=2),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            self.conv_block(512),
            nn.LazyConvTranspose2d(256, kernel_size=2, stride=2),
        )

        self.decoder2 = nn.Sequential(
            self.conv_block(256),
            nn.LazyConvTranspose2d(128, kernel_size=2, stride=2),
        )

        self.decoder3 = nn.Sequential(
            self.conv_block(128),
            nn.LazyConvTranspose2d(64, kernel_size=2, stride=2),
        )

        self.decoder4 = self.conv_block(64)

        # Final layer
        self.final_layer = nn.Sequential(
            nn.LazyConv2d(self.output_channels, kernel_size=1),
            nn.Sigmoid(),  # to ensure the output is in the range of [0, 1]
        )

    def conv_block(self, out_channels):
        return nn.Sequential(
            ResidualBlock(out_channels, use_1x1conv=True),
        )

    def forward(self, X):
        encoder1 = self.encoder1(X)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)

        bottleneck = self.bottleneck(encoder4)

        decoder1 = self.decoder1(torch.cat((encoder4, bottleneck), dim=1))
        decoder2 = self.decoder2(torch.cat((encoder3, decoder1), dim=1))
        decoder3 = self.decoder3(torch.cat((encoder2, decoder2), dim=1))
        decoder4 = self.decoder4(torch.cat((encoder1, decoder3), dim=1))

        return self.final_layer(decoder4)


class RGBD_UNet(nn.Module):
    def __init__(self):
        super(RGBD_UNet, self).__init__()
        self.UNet_R = UNet(output_channels=2)
        self.UNet_G = UNet(output_channels=2)
        self.UNet_B = UNet(output_channels=2)

    def forward(self, RGBD):
        R = self.UNet_R(RGBD[:, [0, 3]])
        G = self.UNet_G(RGBD[:, [1, 3]])
        B = self.UNet_B(RGBD[:, [2, 3]])

        RGB_amp = torch.cat((R[:, :1], G[:, :1], B[:, :1]), dim=1)
        RGB_phs = torch.cat((R[:, 1:], G[:, 1:], B[:, 1:]), dim=1)

        return torch.cat((RGB_amp, RGB_phs), dim=1)


class FourierBlock(nn.Module):
    def __init__(self, num_channels):
        super(FourierBlock, self).__init__()
        self.spatial_conv = ResidualBlock(num_channels, use_1x1conv=True)
        self.fourier_conv = ResidualBlock(num_channels, use_1x1conv=True)

    def forward(self, X):
        spatial_part = self.spatial_conv(X)
        fourier_part = torch.fft.ifft(self.fourier_conv(torch.fft.fft(X)))
        return spatial_part + fourier_part


class Unet_Fourier(UNet):
    def __init__(self, output_channels=6):
        super().__init__(output_channels)

    def conv_block(self, out_channels):
        return FourierBlock(out_channels)
