import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from .utilities import try_gpu
from .bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_single_fixed_distance as BLASM_v3,
)

# BLASM_v1 denotes the class in optics.py,
# BLASM_v2 denotes the parent class bandLimitedAngularSpectrumMethod in bandlimited_angular_spectrum_approach.py,
# BLASM_v3 denotes the child class bandLimitedAngularSpectrumMethod_for_single_fixed_distance in bandlimited_angular_spectrum_approach.py

from .neural_network_components import (
    UNet_imgDepth2AP_v2,
    UNet_imgDepth2AP_heavyweight_v2,
    ResNet_POH,
)

from .perceptual_loss import perceptual_loss


class watermelon_v2(nn.Module):
    def __init__(
        self,
        input_shape,
        perceptual_model_path,
        propagation_distance=torch.tensor([1e-3]),
        heavyweight_UNet=False,  # use UNet_imgDepth2AP by default other than UNet_imgDepth2AP_heavyweight in watermelon_v1
        cuda=True,
    ):
        super(watermelon_v2, self).__init__()

        self.input_shape = input_shape
        self.propagation_distance = propagation_distance

        # load the pre-trained perceptual model
        self.perceptual_model = perceptual_loss(
            input_shape=(
                1,
                6,  # 3 channels of amplitude + 3 channels of phase
                input_shape[-2],
                input_shape[-1],
            ),
            cuda=cuda,
        )
        self.perceptual_model.load_state_dict(torch.load(perceptual_model_path))
        self.perceptual_model.eval()
        self.perceptual_model.requires_grad_(False)

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
        if heavyweight_UNet:
            self.part1 = UNet_imgDepth2AP_heavyweight_v2(output_channels=6).to(self.device)
        else:
            self.part1 = UNet_imgDepth2AP_v2(output_channels=6).to(self.device)

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
        intensity_phs = self.propagator.propagate_P2IP(phs_0)
        # print(f"intensity_phs shape is {intensity_phs.shape}")
        return intensity_phs
        # 6 channels = 3 channels of intensity + 3 channels of phase

    def train_model(
        self,
        train_iter,
        test_iter,
        hyperparameter_lambda=1.0,
        num_epochs=20,
        lr=5e-3,
        milestones=[7, 14],
        hyperparameter_gamma=0.1,
    ):
        model = self
        model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=milestones, gamma=hyperparameter_gamma
        )
        for epoch in range(num_epochs):
            model.train()
            train_loss, n_train = 0.0, 0
            # 4 channels = 3 channels of intensity + 1 channel of depth
            for img_depth in train_iter:

                y_hat = model(
                    img_depth
                )  # 6 channels = 3 channels of intensity + 3 channels of phase

                perceptual_model_input = torch.cat(
                    (torch.sqrt(img_depth[:, :3]), y_hat[:, 3:]), dim=1
                )  # 6 channels = 3 channels of amplitude + 3 channels of phase

                l = self.loss(
                    y_hat[:, :3], img_depth[:, :3]
                ) + hyperparameter_lambda * self.loss(
                    self.perceptual_model(perceptual_model_input), img_depth[:, 3:]
                )

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                train_loss += l.item()
                n_train += img_depth.size(0)

            model.eval()
            test_loss, n_test = 0.0, 0
            for img_depth in test_iter:
                with torch.no_grad():
                    y_hat = model(img_depth)

                    perceptual_model_input = torch.cat(
                        (torch.sqrt(img_depth[:, :3]), y_hat[:, 3:]), dim=1
                    )

                    l = self.loss(
                        y_hat[:, :3], img_depth[:, :3]
                    ) + hyperparameter_lambda * self.loss(
                        self.perceptual_model(perceptual_model_input), img_depth[:, 3:]
                    )

                test_loss += l.item()
                n_test += img_depth.size(0)

            self.scheduler.step()

            print(
                f"epoch {epoch + 1}, train loss {train_loss / n_train:.4f}, test loss {test_loss / n_test:.4f}"
            )

    def loss(self, y_hat, y):
        loss = nn.MSELoss()
        return loss(y_hat, y)

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
