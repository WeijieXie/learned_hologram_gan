import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utilities import try_gpu, generate_checkerboard_mask, amplitude_normalizor

from ..bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_single_fixed_distance as fixed_distance_propogator,
)
from ..neural_network_components import miniResNet, ChannelWiseSymmetricConv

from .loss_func import amp_loss


class AP2POH(nn.Module):
    def __init__(
        self,
        input_shape=(1, 6, 192, 192),
        pretrained_model_path=None,
        freeze=False,
        cuda=True,
        pad_size=192,
        filter_radius_coefficient=0.5,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([638e-9, 520e-9, 450e-9]),
        distance=torch.tensor([1e-3]),
    ):
        super(AP2POH, self).__init__()

        self.input_shape = input_shape
        self.pretrained_model_path = pretrained_model_path
        self.freeze = freeze
        self.device = try_gpu() if cuda else torch.device("cpu")

        self.checkerboard_mask_1 = generate_checkerboard_mask(
            input_shape[-2],
            input_shape[-1],
            1,
            True,
        ).to(self.device)

        self.checkerboard_mask_2 = generate_checkerboard_mask(
            input_shape[-2],
            input_shape[-1],
            1,
            False,
        ).to(self.device)

        self.propagator = fixed_distance_propogator(
            sample_row_num=input_shape[-2],
            sample_col_num=input_shape[-1],
            pad_size=pad_size,
            filter_radius_coefficient=filter_radius_coefficient,
            pixel_pitch=pixel_pitch,
            wave_length=wave_length,
            # pad_size=192,
            # pixel_pitch=3.74e-6,
            # wave_length=torch.tensor([638e-9, 520e-9, 450e-9]),
            band_limit=False,
            cuda=cuda,
            distance=distance,
            # distance=torch.tensor([1e-3]),
        )

        self.part1 = ChannelWiseSymmetricConv(kernel_size=3, padding=1).to(self.device)
        # self.filter_radius_coefficient = nn.Parameter(
        #     torch.tensor(
        #         [filter_radius_coefficient], requires_grad=True, device=self.device
        #     ),
        #     requires_grad=True,
        # )

        self._initialize_weights()

        if self.pretrained_model_path is not None:
            self.load_state_dict(torch.load(self.pretrained_model_path))
            if freeze:
                self.eval()
                self.requires_grad_(False)
        # else:
        #     self.filter_radius_coefficient.requires_grad_(False)

    def dataloader_filter(self, amp, phs, filter_radius_coefficient):
        g_filtered = self.propagator.cropping(
            torch.fft.ifft2(
                torch.fft.fft2(self.propagator.padding(amp * torch.exp(1j * phs)))
                * self.propagator.generate_circular_frequency_mask_differentiable(
                    filter_radius_coefficient
                )
            )
        )
        return torch.abs(g_filtered), torch.angle(g_filtered)

    def double_phase_method(self, amp, phs):
        acos_amp = torch.acos(amp)

        # phs_2pi = 2 * torch.pi * phs

        phs_1 = phs + acos_amp
        phs_2 = phs - acos_amp

        POH = self.checkerboard_mask_1 * phs_1 + self.checkerboard_mask_2 * phs_2

        return POH

    def phs_sincos(self, phs):

        sin_phs = torch.sin(phs)
        cos_phs = torch.cos(phs)

        return torch.cat((sin_phs, cos_phs), dim=-3)

    def forward(self, amp_z, phs_z):

        # complex_field = self.propagator.propagate_AP2C_backward(amp_z, phs_z)
        # part1_input = torch.cat(
        #     (torch.real(complex_field), torch.imag(complex_field)), dim=-3
        # )
        # part1_output = self.part1(part1_input) - 0.5
        # modified_complex_field = (
        #     torch.complex(part1_output[:, :3], part1_output[:, 3:]) + complex_field
        # )
        # # Double phase method
        # POH = self.double_phase_method(
        #     amplitude_normalizor(torch.abs(modified_complex_field)),
        #     torch.angle(modified_complex_field),
        # )
        # return POH

        # amplitude modulation
        complex_field = self.propagator.propagate_AP2C_backward(amp_z, phs_z)
        modified_complex_field = torch.complex(
            self.part1(torch.real(complex_field)), self.part1(torch.imag(complex_field))
        )
        # Double phase method
        POH = self.double_phase_method(
            amplitude_normalizor(torch.abs(modified_complex_field)),
            torch.angle(modified_complex_field),
        )
        return POH

    def train_model(
        self,
        train_loader,
        val_loader,
        filter_radius_coefficient=0.45,
        epochs=30,
        lr=1e-3,
        alpha=1e-3,
        beta=1e-5,
        hyperparameter_gamma=0.1,
        save_path=None,
        checkpoint_iterval=10,
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

        xxx = 0  #####################################################################

        for epoch in range(epochs):

            # train
            model.train()
            train_loss, n_train = 0.0, 0

            for amp, phs in train_loader:
                amp, phs = self.dataloader_filter(amp, phs, filter_radius_coefficient)

                phs_hat = model(amp, phs)
                amp_hat, phs_hat, spectrum_loss = (
                    self.propagator.propagate_POH2AP_forward_with_spectrum_loss(
                        phs_hat, filter_radius_coefficient
                    )
                )
                l = self.loss(amp_hat, amp, alpha) + beta * spectrum_loss

                self.optimizer.zero_grad()
                l.backward()
                # xxx = self.filter_radius_coefficient.grad.item()  #####################
                self.optimizer.step()

                train_loss += l.item()
                n_train += phs_hat.size(0)

            # validation
            model.eval()
            test_loss, n_test = 0.0, 0
            for amp, phs in val_loader:
                amp, phs = self.dataloader_filter(amp, phs, filter_radius_coefficient)

                with torch.no_grad():
                    phs_hat = model(amp, phs)
                    amp_hat, phs_hat, spectrum_loss = (
                        self.propagator.propagate_POH2AP_forward_with_spectrum_loss(
                            phs_hat, filter_radius_coefficient
                        )
                    )
                    l = self.loss(amp_hat, amp, alpha) + beta * spectrum_loss

                test_loss += l.item()
                n_test += phs_hat.size(0)

            average_train_loss = train_loss / n_train
            average_test_loss = test_loss / n_test
            self.train_loss.append(average_train_loss)
            self.test_loss.append(average_test_loss)
            print(
                f"epoch {epoch + 1}, train loss {average_train_loss:.7f}, test loss {average_test_loss:.7f},filter_radius_coefficient {filter_radius_coefficient.item()},gradient {xxx}"
            )

            # update learning rate
            self.scheduler.step(average_test_loss)

            # checkpoint
            if epoch % checkpoint_iterval == 0 and epoch != 0 and save_path is not None:
                check_point_path = save_path.replace(".pth", f"_epoch{epoch}.pth")
                torch.save(model.state_dict(), check_point_path)

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    def loss(
        self,
        amp_hat,
        amp,
        alpha,
    ):
        return amp_loss(
            amp_hat,
            amp,
            alpha,
        )

    def _initialize_weights(self):
        # Initialize weights by running a dummy forward pass
        dummy_input = torch.randn(*self.input_shape).to(self.device)
        _ = self.part1(torch.abs(dummy_input))

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
