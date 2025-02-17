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


class UNet_encoder(nn.Module):
    def __init__(
        self,
    ):
        super(UNet_encoder, self).__init__()

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

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(1024),
        )

    def forward(self, X):
        encoder1 = self.encoder1(X)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder_bottleneck = self.bottleneck(encoder4)

        return encoder1, encoder2, encoder3, encoder4, encoder_bottleneck

    def conv_block(self, out_channels):
        return nn.Sequential(
            ResidualBlock(out_channels, use_1x1conv=True),
        )


class UNet_decoder(nn.Module):
    def __init__(
        self,
        output_channels=3,
    ):
        super(UNet_decoder, self).__init__()

        self.bottleneck = nn.LazyConvTranspose2d(512, kernel_size=2, stride=2)

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

        self.final_layer = nn.Sequential(
            nn.LazyConv2d(output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        encoder1_output,
        encoder2_output,
        encoder3_output,
        encoder4_output,
        encoder_bottleneck_output,
    ):
        decoder_bottleneck_output = self.bottleneck(encoder_bottleneck_output)
        decoder1 = self.decoder1(
            torch.cat((encoder4_output, decoder_bottleneck_output), dim=1)
        )
        decoder2 = self.decoder2(torch.cat((encoder3_output, decoder1), dim=1))
        decoder3 = self.decoder3(torch.cat((encoder2_output, decoder2), dim=1))
        decoder4 = self.decoder4(torch.cat((encoder1_output, decoder3), dim=1))

        return self.final_layer(decoder4)

    def conv_block(self, out_channels):
        return nn.Sequential(
            ResidualBlock(out_channels, use_1x1conv=True),
        )


class Unet_decoder_with_attention(UNet_decoder):
    def __init__(self, output_channels=3):
        super(Unet_decoder_with_attention, self).__init__(output_channels)

        self.attention_block0 = attention_block(1024)
        self.attention_block1 = attention_block(512)
        self.attention_block2 = attention_block(256)
        self.attention_block3 = attention_block(128)
        self.attention_block4 = attention_block(64)

        # sometimes the amplitude may exceed 1
        self.amp_coefficient = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(
        self,
        encoder1_output,
        encoder2_output,
        encoder3_output,
        encoder4_output,
        encoder_bottleneck_output,
        distances=torch.tensor([0.0]),
    ):
        batch_size = encoder_bottleneck_output.shape[0]
        distances_num = distances.shape[0]
        new_batch_size = batch_size * distances_num

        bottleneck_size = encoder_bottleneck_output.shape[-1]
        decoder1_size = bottleneck_size * 2
        decoder2_size = decoder1_size * 2
        decoder3_size = decoder2_size * 2
        decoder4_size = decoder3_size * 2

        # calculate the attention weights which is ditermined by the distances
        # shape: (distances_num, channel_weights, 1, 1)
        bottleneck_attention = self.attention_block0(distances)
        decoder1_attention = self.attention_block1(distances)
        decoder2_attention = self.attention_block2(distances)
        decoder3_attention = self.attention_block3(distances)
        decoder4_attention = self.attention_block4(distances)

        # add weights to the 1024 channels
        # shape: (batch_size, distances_num, 1024, bottleneck_size, bottleneck_size)
        encoder_bottleneck_output_with_attention = (
            encoder_bottleneck_output.unsqueeze(1) * bottleneck_attention
        )
        # by ConvTranspose2d, the channels is cut half and size is doubled
        # shape: (batch_size * distances_num, 512, decoder1_size, decoder1_size)
        decoder_bottleneck_output = self.bottleneck(
            encoder_bottleneck_output_with_attention.view(
                new_batch_size, -1, bottleneck_size, bottleneck_size
            )
        )

        # shape: (batch_size, distances_num, 512, decoder1_size, decoder1_size)
        decoder_bottleneck_output_with_attention = (
            decoder_bottleneck_output.view(
                batch_size, distances_num, -1, decoder1_size, decoder1_size
            )
            * decoder1_attention
        )
        decoder1 = self.decoder1(
            # shape: (batch_size * distances_num, 512+512, decoder1_size, decoder1_size)
            torch.cat(
                (
                    encoder4_output.repeat(distances_num, 1, 1, 1),
                    decoder_bottleneck_output_with_attention.view(
                        new_batch_size, -1, decoder1_size, decoder1_size
                    ),
                ),
                dim=-3,
            )
        )

        decoder1_with_attention = (
            decoder1.view(batch_size, distances_num, -1, decoder2_size, decoder2_size)
            * decoder2_attention
        )
        decoder2 = self.decoder2(
            torch.cat(
                (
                    encoder3_output.repeat(distances_num, 1, 1, 1),
                    decoder1_with_attention.view(
                        new_batch_size, -1, decoder2_size, decoder2_size
                    ),
                ),
                dim=-3,
            )
        )

        decoder2_with_attention = (
            decoder2.view(batch_size, distances_num, -1, decoder3_size, decoder3_size)
            * decoder3_attention
        )
        decoder3 = self.decoder3(
            torch.cat(
                (
                    encoder2_output.repeat(distances_num, 1, 1, 1),
                    decoder2_with_attention.view(
                        new_batch_size, -1, decoder3_size, decoder3_size
                    ),
                ),
                dim=-3,
            )
        )

        decoder3_with_attention = (
            decoder3.view(batch_size, distances_num, -1, decoder4_size, decoder4_size)
            * decoder4_attention
        )
        decoder4 = self.decoder4(
            torch.cat(
                (
                    encoder1_output.repeat(distances_num, 1, 1, 1),
                    decoder3_with_attention.view(
                        new_batch_size, -1, decoder4_size, decoder4_size
                    ),
                ),
                dim=-3,
            )
        )

        return self.final_layer(decoder4) * self.amp_coefficient


class attention_block(nn.Module):
    """
    Attention block for multi-head UNet

    Args:
        output_dim: int, the output dimension of the attention block

    Returns:
        X: tensor, the output of the attention block with shape
    """

    def __init__(self, output_dim=None):
        super(attention_block, self).__init__()
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, X):
        X = self.net(X.view(-1, 1))
        return X.unsqueeze(-1).unsqueeze(-2)


class multi_head_UNet(nn.Module):
    def __init__(self, output_channels=(12, 3)):
        super(multi_head_UNet, self).__init__()

        self.output_channels = output_channels

        self.encoder = UNet_encoder()

        self.decoder1 = UNet_decoder(output_channels[0])
        self.decoder2 = Unet_decoder_with_attention(output_channels[1])

    def forward(self, X, distances):

        encoder1, encoder2, encoder3, encoder4, bottleneck = self.encoder(X)

        return (
            self.decoder1(encoder1, encoder2, encoder3, encoder4, bottleneck),
            self.decoder2(
                encoder1, encoder2, encoder3, encoder4, bottleneck, distances
            ),
        )


class multi_head_UNet_CGH(nn.Module):
    def __init__(self, output_channels=(3, 3)):
        super(multi_head_UNet_CGH, self).__init__()

        self.output_channels = output_channels

        self.encoder = UNet_encoder()

        self.decoder1 = UNet_decoder(output_channels[0])
        self.decoder2 = UNet_decoder(output_channels[1])

    def forward(self, X):

        encoder1, encoder2, encoder3, encoder4, bottleneck = self.encoder(X)

        return (
            self.decoder1(encoder1, encoder2, encoder3, encoder4, bottleneck),
            self.decoder2(encoder1, encoder2, encoder3, encoder4, bottleneck),
        )


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


class UNet_CGH(UNet):
    def __init__(self, output_channels=6):
        super(UNet_CGH, self).__init__(output_channels)

    def forward(self, X):
        result = 2 * torch.pi * super(UNet_CGH, self).forward(X)
        return result[:, :3, :, :], result[:, 3:, :, :]


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


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        # Encoder: outputs both mean and log variance
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)  # Mean layer
        self.fc_logvar = nn.Linear(400, 20)  # Log variance layer

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Sample epsilon from N(0,1)
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))  # Flatten input image
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def vae_loss(recon_x, x, mu, logvar):
        # Reconstruction loss (Binary Cross-Entropy)
        bce_loss = nn.functional.binary_cross_entropy(
            recon_x, x.view(-1, 784), reduction="sum"
        )

        # KL Divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        return bce_loss + kld_loss
