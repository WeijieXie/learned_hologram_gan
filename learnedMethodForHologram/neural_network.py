import torch
from torch import nn
from torch.nn import functional as F

import itertools

from .utilities import try_gpu
from .bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_single_fixed_distance as BLASM_v3,
)

# BLASM_v1 denotes the class in optics.py,
# BLASM_v2 denotes the parent class bandLimitedAngularSpectrumMethod in bandlimited_angular_spectrum_approach.py,
# BLASM_v3 denotes the child class bandLimitedAngularSpectrumMethod_for_single_fixed_distance in bandlimited_angular_spectrum_approach.py


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


class ResNet_POH(nn.Module):
    def __init__(self, output_channels=3):
        super(ResNet_POH, self).__init__()
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
        return 2 * torch.pi * self.net(X)


class ResNet_FashionMnist(nn.Module):
    def __init__(self, shape):
        super(ResNet_FashionMnist, self).__init__()
        self.input_shape = shape
        self.net = nn.Sequential(
            self.part_1(),  # conv and pooling
            self.part_2(),  # residual blocks
            self.part_3(),  # global pooling and dense layer
        )
        self._initialize_weights()
        self.device = try_gpu()
        self.to(self.device)

    def part_1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            # nn.LazyConv2d(32, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def part_2(self):
        return nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(128, use_1x1conv=True, strides=2),
            ResidualBlock(128),
            ResidualBlock(256, use_1x1conv=True, strides=2),
            ResidualBlock(256),
            ResidualBlock(512, use_1x1conv=True, strides=2),
            ResidualBlock(512),
            #     ResidualBlock(32),
            #     ResidualBlock(32),
            #     ResidualBlock(64, use_1x1conv=True, strides=2),
            #     ResidualBlock(64),
            #     ResidualBlock(128, use_1x1conv=True, strides=2),
            #     ResidualBlock(128),
            #     ResidualBlock(256, use_1x1conv=True, strides=2),
            #     ResidualBlock(256),
        )

    def part_3(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10),
            # nn.Softmax(dim=1), # can be removed if loss function is CrossEntropyLoss
        )

    def _initialize_weights(self):
        # Initialize weights by running a dummy forward pass
        dummy_input = torch.randn(*self.input_shape)
        _ = self.forward(dummy_input)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LazyBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.LazyLinear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        return self.net(X)

    def train_model(self, train_iter, test_iter, num_epochs, lr, device):
        model = self
        model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            train_loss, train_accuracy, n = 0.0, 0.0, 0
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                l = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_loss += l
                train_accuracy += (y_hat.argmax(axis=1) == y).sum()
                n += y.size(0)
            test_accuracy = self.evaluate_accuracy(test_iter, model)
            print(
                f"epoch {epoch + 1}, "
                f"train loss {train_loss / n:.4f}, "
                f"train accuracy {train_accuracy / n:.3f}, "
                f"test accuracy {test_accuracy:.3f}"
            )

    def evaluate_accuracy(self, data_iter, net):
        net.eval()
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = net(X)
            acc_sum += (y_hat.argmax(axis=1) == y).sum()
            n += y.size(0)
        net.train()
        return acc_sum / n

    def loss(self, y_hat, y):
        evaluator = nn.CrossEntropyLoss()
        return evaluator(y_hat, y)


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
            nn.Sigmoid(),
        )

    def conv_block(self, out_channels):
        return nn.Sequential(
            ResidualBlock(out_channels, use_1x1conv=True),
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

        return 2 * torch.pi * self.final_layer(decoder4)


class watermelon(nn.Module):
    def __init__(self, input_shape, propagation_distance, cuda=True):
        super(watermelon, self).__init__()

        self.input_shape = input_shape
        self.propagation_distance = propagation_distance
        self.device = try_gpu() if cuda else torch.device("cpu")

        # propagator
        self.propagator = BLASM_v3(
            sample_row_num=192,
            sample_col_num=192,
            pixel_pitch=3.74e-6,
            wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
            band_limit=False,
            cuda=cuda,
            distance=self.propagation_distance,
        )

        # a UNet used for generate amp and phs from rgbd input
        self.part1 = UNet(output_channels=6).to(self.device)

        # a ResNet (without pooling) used for generate phase-only hologram from amp and phs
        self.part2 = ResNet_POH(output_channels=3).to(self.device)

        # Initialize weights
        self._initialize_weights()

    def forward(self, X):
        # print(f"X shape is {X.shape}")
        amp_phs_z = self.part1(X)
        # print(f"amp_phs_z shape is {amp_phs_z.shape}")
        amp_phs_0 = self.propagator.propagate_AP2AP(amp_phs_z)
        # print(f"amp_phs_0 shape is {amp_phs_0.shape}")
        phs_0 = self.part2(amp_phs_0)
        # print(f"phs_0 shape is {phs_0.shape}")
        intensity = self.propagator.propagate_P2I(phs_0)
        # print(f"intensity shape is {intensity.shape}")
        return intensity

    def train_model(self, train_iter, test_iter, num_epochs, lr):
        model = self
        model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            model.train()
            train_loss, n_train = 0.0, 0
            for img_depth in train_iter:
                y_hat = model(img_depth)
                l = self.loss(y_hat, img_depth[:, :3])
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                train_loss += l.item()
                n_train += img_depth.size(0)

            self.eval()
            test_loss, n_test = 0.0, 0
            for img_depth in test_iter:
                with torch.no_grad():
                    y_hat = model(img_depth)
                l = self.loss(y_hat, img_depth[:, :3])

                test_loss += l.item()
                n_test += img_depth.size(0)
            print(
                f"epoch {epoch + 1}, train loss {train_loss / n_train:.4f}, test loss {test_loss / n_test:.4f}"
            )

    def loss(self, y_hat, y):
        loss = nn.MSELoss()
        return loss(y_hat, y)

    def _initialize_weights(self):
        # Initialize weights by running a dummy forward pass
        dummy_input = torch.randn(*self.input_shape).to(self.device)
        _ = self.forward(dummy_input)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.ConvTranspose2d, nn.LazyConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LazyBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.LazyLinear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
