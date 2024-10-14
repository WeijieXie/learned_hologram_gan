import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utilities import try_gpu

from .RGBD2AP import RGBD2AP
from .AP2POH import AP2POH
from ..angular_spectrum_method import (
    bandLimitedAngularSpectrumMethod_for_multiple_distances as multiple_distances_propogator,
)


class Generator(nn.Module):
    def __init__(
        self,
        sample_row_num=192,
        sample_col_num=192,
        pad_size=160,
        filter_radius_coefficient=0.5,
        kernel_size=3,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([638e-9, 520e-9, 450e-9]),
        distance=torch.tensor([1e-3]),
        pretrained_model_path=None,
        pretrained_model_path_RGBD2AP=None,
        pretrained_model_path_AP2POH=None,
    ):
        super(Generator, self).__init__()

        self.part1 = RGBD2AP(
            input_shape=(1, 4, sample_row_num, sample_col_num),
            pretrained_model_path=pretrained_model_path_RGBD2AP,
            freeze=False,
            cuda=True,
            amplitude_scaler=1.1,
        )

        self.part2 = AP2POH(
            input_shape=(1, 6, sample_row_num, sample_col_num),
            pretrained_model_path=pretrained_model_path_AP2POH,
            freeze=False,
            cuda=True,
            filter_radius_coefficient=filter_radius_coefficient,
            pad_size=pad_size,
            pixel_pitch=pixel_pitch,
            wave_length=wave_length,
            distance=distance,
            kernel_size=kernel_size,
        )

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path))

    def forward(self, RGBD):
        amp_hat, phs_hat = self.part1(RGBD)
        phs_hat = self.part2(amp_hat, phs_hat)
        return phs_hat
