import torch
from torch import nn
from torch.nn import functional as F

from .utilities import try_gpu
from .bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_single_fixed_distance as BLASM_v3,
)

from .neural_network_components import UNet, Unet_Fourier, ResidualBlock


class perceptual_loss(nn.Module):
    """
    A class to define the perceptual loss function, which learns the relationship between the complex amplitude of the hologram and the depth map.
    The model takes 3 channels (complex amplitude of the rgb) and returns the depth map.
    """

    def __init__(
        self,
        input_shape=(1, 6, 192, 192),  # the amplitude and phase of the rgb image
        cuda=True,
    ):
        super(perceptual_loss, self).__init__()

        self.input_shape = input_shape
        self.device = try_gpu() if cuda else torch.device("cpu")

        # self.part1 = Unet_Fourier(output_channels=32).to(self.device)
        self.part1 = UNet(output_channels=32).to(self.device)

        self.part2 = nn.Sequential(
            ResidualBlock(num_channels=3, use_1x1conv=True),
            nn.LazyConv2d(out_channels=1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        ).to(self.device)

        # Initialize weights
        self._initialize_weights()

    def forward(self, X):
        return self.part2(self.part1(X))

    def loss(self, y_hat, y):
        loss = nn.MSELoss()
        return loss(y_hat, y)

    def train_model(self, train_iter, test_iter, num_epochs, lr):
        """
        takes in the amplitude and phase of the rgb image and trains the model
        """
        model = self
        model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):

            # train
            model.train()
            train_loss, n_train = 0.0, 0
            for img3ch_phs3ch_depth in train_iter:

                y_hat = model(img3ch_phs3ch_depth[:,:-1])
                l = self.loss(y_hat, img3ch_phs3ch_depth[:,-1].unsqueeze(1))

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                train_loss += l.item()
                n_train += img3ch_phs3ch_depth.size(0)

            # test
            self.eval()
            test_loss, n_test = 0.0, 0

            for img3ch_phs3ch_depth in test_iter:

                with torch.no_grad():
                    y_hat = model(img3ch_phs3ch_depth[:,:-1])
                l = self.loss(y_hat, img3ch_phs3ch_depth[:,-1].unsqueeze(1))

                test_loss += l.item()
                n_test += img3ch_phs3ch_depth.size(0)

            print(
                f"epoch {epoch + 1}, train loss {train_loss / n_train:.4f}, test loss {test_loss / n_test:.4f}"
            )

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
