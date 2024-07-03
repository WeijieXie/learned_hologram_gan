import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities


class bandLimitedAngularSpectrumMethod:
    def __init__(
        self,
        amplitudeTensor,
        phaseTensor,
        pixel_pitch,
        distances,
        padding=False,
        device="cpu",
    ):
        if not isinstance(amplitudeTensor, torch.Tensor) or not isinstance(
            phaseTensor, torch.Tensor
        ):
            raise ValueError("Amplitude tensor or phase tensor is required")
        if amplitudeTensor.shape != phaseTensor.shape:
            raise ValueError("Amplitude and phase tensors must have the same shape")

        if device == "cuda":
            device = utilities.try_gpu()
        self.device = device

        self.amplitudeTensor = amplitudeTensor.to(device)
        self.phaseTensor = phaseTensor.to(device)
        self.pixel_pitch = pixel_pitch
        self.distances = distances

        self.samplingRowNum = amplitudeTensor.shape[-2]
        self.samplingColNum = amplitudeTensor.shape[-1]
        self.padding = padding

    def frequencyMesh(self):
        S_x = sample_u * dx  # the x-size of the hologram plain
        S_y = sample_v * dx  # the y-size of the hologram plain

    def band_limited_angular_spectrum_multichannels(
        self,
        amplitude=None,
        phase=None,
        propagation_distance=torch.Tensor([0.0]),
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        pixel_pitch=3.74e-6,
        band_limit=True,
        debug=False,
    ):

        dx = pixel_pitch  # the sampling interval
        sample_u = amplitude.shape[1]
        sample_v = amplitude.shape[2]
        S_x = sample_u * dx  # the x-size of the hologram plain
        S_y = sample_v * dx  # the y-size of the hologram plain

        # generate the 2-D frequency mesh
        freq_x = torch.fft.fftfreq(sample_u, dx)
        freq_y = torch.fft.fftfreq(sample_v, dx)
        if debug:
            print(
                "The highest positive frequency is u = {} and v = {}.".format(
                    freq_x.max(), freq_y.max()
                )
            )
            print(
                "The highest negative frequency is u = {} and v = {}.".format(
                    freq_x.min(), freq_y.min()
                )
            )
            print(
                "The resolution of the frequency is u = {} and v = {}.".format(
                    freq_x[1] - freq_x[0], freq_y[1] - freq_y[0]
                )
            )

        freq_x_unsqueezed = freq_x.unsqueeze(1).expand(sample_u, sample_v)
        freq_y_unsqueezed = freq_y.unsqueeze(0).expand(sample_u, sample_v)

        freq_square = freq_x_unsqueezed**2 + freq_y_unsqueezed**2
        freq_cube = freq_square.unsqueeze(0).repeat(3, 1, 1)

        freq_max = 1 / wave_length**2
        freq_max_cube = freq_max.unsqueeze(1).unsqueeze(2).repeat(1, sample_u, sample_v)

        w_cube_0 = freq_max_cube - freq_cube
        mask_w_cube = w_cube_0 > 0
        w_cube = mask_w_cube * w_cube_0

        # transfer function
        H_FR = torch.exp(2j * math.pi * propagation_distance * torch.sqrt(w_cube))
        # H_FR = torch.exp(1j * math.pi * z * (2/wave_length.unsqueeze(1).unsqueeze(2)-wave_length.unsqueeze(1).unsqueeze(2)*freq_cube))
        print(H_FR.shape)

        if band_limit:
            # clipper the frequency
            d_u = 1 / S_x  # S_x instead of 2 * S_x
            d_v = 1 / S_y  # S_y instead of 2 * S_y
            u_limit = 1 / (math.sqrt((2 * d_u * propagation_distance) ** 2 + 1) * wave_length)
            v_limit = 1 / (math.sqrt((2 * d_v * propagation_distance) ** 2 + 1) * wave_length)
            mask_u = torch.abs(freq_x).unsqueeze(0).repeat(3, 1) < u_limit.unsqueeze(
                1
            ).expand(3, sample_u)
            mask_v = torch.abs(freq_y).unsqueeze(0).repeat(3, 1) < v_limit.unsqueeze(
                1
            ).expand(3, sample_v)
            mask = mask_u.unsqueeze(2) & mask_v.unsqueeze(1)
            H_FR = H_FR * mask

        sourcePlain = utilities.complex_plain(amplitude, phase)
        G_0 = torch.fft.fft2(sourcePlain)
        G_z = G_0 * H_FR

        # inverse fourier transform
        g_z_complex = torch.fft.ifft2(G_z)

        if debug:
            if mask.sum() == sample_u * sample_v * 3:
                print(
                    "The maximum frequency is clipped to u = {} and v = {}.".format(
                        u_limit, v_limit
                    )
                )
                print("The clipper is NOT working............")
            else:
                print(
                    "The maximum frequency is clipped to u = {} and v = {}.".format(
                        u_limit, v_limit
                    )
                )
                print("The clipper is working............")

        return g_z_complex
