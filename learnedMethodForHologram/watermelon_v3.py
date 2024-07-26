import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from .watermelon_v2 import watermelon_v2

from .bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_single_fixed_distance as BLASM_v3,
)

from .perceptual_loss import perceptual_loss


class watermelon_v3(watermelon_v2):
    def __init__(
        self,
        input_shape,
        perceptual_model_path,
        propagation_distance=torch.tensor([1e-3]),
        heavyweight_UNet=False,  # use UNet_imgDepth2AP by default other than UNet_imgDepth2AP_heavyweight in watermelon_v1
        cuda=True,
    ):
        super(watermelon_v3, self).__init__(
            input_shape,
            perceptual_model_path,
            propagation_distance=propagation_distance,
            heavyweight_UNet=heavyweight_UNet,
            cuda=cuda,
        )

    def forward(self, X):
        # print(f"X shape is {X.shape}")
        amp_phs_z = self.part1(X)
        # print(f"amp_phs_z shape is {amp_phs_z.shape}")
        amp_phs_0 = self.propagator.propagate_AP2AP(amp_phs_z)
        # print(f"amp_phs_0 shape is {amp_phs_0.shape}")
        phs_0 = self.part2(amp_phs_0)
        # print(f"phs_0 shape is {phs_0.shape}")
        amp_phs = self.propagator.propagate_P2IP(phs_0)
        # print(f"intensity_phs shape is {intensity_phs.shape}")
        return amp_phs
        # 6 channels = 3 channels of amp + 3 channels of phase

    def train_model(
        self,
        train_iter,
        test_iter,
        hyperparameter_lambda=1.0,
        num_epochs=20,
        lr=1e-3,
        hyperparameter_gamma=0.1,
        # milestones=[7, 14],
    ):
        model = self
        model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # self.scheduler = MultiStepLR(
        #     self.optimizer, milestones=milestones, gamma=hyperparameter_gamma
        # )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=hyperparameter_gamma,
            patience=2,
            verbose=True,
            threshold=1e-3,
            threshold_mode="rel",
        )

        for epoch in range(num_epochs):
            model.train()
            train_loss, n_train = 0.0, 0
            # 4 channels = 3 channels of intensity + 1 channel of depth
            for img_depth in train_iter:

                y_hat = model(img_depth)
                # 6 channels = 3 channels of amplitude + 3 channels of phase

                l = self.loss(
                    (y_hat[:, :3]) ** 2, img_depth[:, :3]
                ) + hyperparameter_lambda * self.loss(
                    self.perceptual_model(y_hat), img_depth[:, 3:]
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

                    l = self.loss(
                        (y_hat[:, :3]) ** 2, img_depth[:, :3]
                    ) + hyperparameter_lambda * self.loss(
                        self.perceptual_model(y_hat), img_depth[:, 3:]
                    )

                test_loss += l.item()
                n_test += img_depth.size(0)

            print(
                f"epoch {epoch + 1}, train loss {train_loss / n_train:.6f}, test loss {test_loss / n_test:.6f}"
            )

            self.scheduler.step(test_loss / n_test)
            # self.scheduler.step()
