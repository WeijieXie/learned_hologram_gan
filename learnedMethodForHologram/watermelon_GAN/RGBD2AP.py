import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utilities import try_gpu

from ..neural_network_components import (
    UNet,
)

from .loss_func import amp_phs_loss


class RGBD2AP(nn.Module):
    def __init__(
        self,
        input_shape=(1, 4, 192, 192),
        pretrained_model_path=None,
        freeze=False,
        cuda=True,
    ):
        super(RGBD2AP, self).__init__()

        self.input_shape = input_shape
        self.pretrained_model_path = pretrained_model_path
        self.freeze = freeze
        self.device = try_gpu() if cuda else torch.device("cpu")

        # a UNet whose input is RGBD and output is amplitude(0~1) and phase(0~1)
        self.part1 = UNet(output_channels=6).to(self.device)
        self.adjustment_factor = nn.Parameter(torch.tensor([1.0]))

        if self.pretrained_model_path is not None:
            self.part1.load_state_dict(torch.load(self.pretrained_model_path))
            if freeze:
                self.part1.eval()
                self.part1.requires_grad_(False)
        else:
            self._initialize_weights()

    def forward(self, RGBD):
        """
        take the 4 channels of RGBD as input and output the 6 channels of amplitude and phase
        """
        y = self.part1(RGBD)
        y[:, :3, :, :] = self.adjustment_factor * y[:, :3, :, :]  # amplitude
        y[:, 3:, :, :] = 2 * torch.pi * y[:, 3:, :, :]  # phase
        return y

    def train_model(
        self,
        train_loader,
        val_loader,
        epochs=30,
        lr=1e-3,
        alpha=1e-3,
        hyperparameter_gamma=0.1,
        save_path=None,
    ):
        if self.freeze:
            raise ValueError("The model is frozen, cannot be trained")

        if save_path is None:
            print(
                "!!!!!!The save path is not specified, the model will not be saved!!!!!!"
            )

        if self.pretrained_model_path is not None:
            print("The model is pretrained, will be fine-tuned or continued training")

        self.train_loss = []
        self.test_loss = []

        model = self
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=hyperparameter_gamma,
            patience=4,
            verbose=True,
            threshold=1e-3,
            threshold_mode="rel",
            min_lr=1e-6,
        )

        for epoch in range(epochs):

            # train
            model.train()
            train_loss, n_train = 0.0, 0
            for amp, phs, img_depth in train_loader:

                amp_phs_hat = model(img_depth)

                l = self.loss(amp_phs_hat, amp, phs, alpha)

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                train_loss += l.item()
                n_train += img_depth.size(0)

            # validation
            model.eval()
            test_loss, n_test = 0.0, 0
            for amp, phs, img_depth in val_loader:

                with torch.no_grad():
                    amp_phs_hat = model(img_depth)
                    l = self.loss(amp_phs_hat, amp, phs, alpha)

                test_loss += l.item()
                n_test += img_depth.size(0)

            average_train_loss = train_loss / n_train
            average_test_loss = test_loss / n_test
            self.train_loss.append(average_train_loss)
            self.test_loss.append(average_test_loss)
            print(
                f"epoch {epoch + 1}, train loss {average_train_loss:.7f}, test loss {average_test_loss:.7f}"
            )

            # update learning rate
            self.scheduler.step(average_test_loss)

            # checkpoint
            if epoch % 10 == 0 and epoch != 0 and save_path is not None:
                check_point_path = save_path.replace(".pth", f"_epoch{epoch}.pth")
                torch.save(model.state_dict(), check_point_path)

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    def loss(
        self,
        amp_phs_hat,
        amp,
        phs,
        alpha,
    ):
        return amp_phs_loss(
            amp_phs_hat,
            2 * torch.pi * amp,
            2 * torch.pi * phs,
            alpha,
        )

    def _initialize_weights(self):
        # Initialize weights by running a dummy forward pass
        dummy_input = torch.randn(*self.input_shape).to(self.device)
        _ = self.forward(torch.abs(dummy_input))

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
