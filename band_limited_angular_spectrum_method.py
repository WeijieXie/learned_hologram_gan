import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities


class bandLimitedAngularSpectrumMethod:
    def __init__(
        self,
        amplitudeTensor=None,
        phaseTensor=None,
        distances=torch.Tensor([0.0]),
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        padding=False,
        debug=False,
        device="cpu",
    ):
        if amplitudeTensor is None:
            self.amplitudeTensor = (
                utilities.amplitude_tensor_generator_for_phase_only_hologram(
                    ".\data\images\sample_hologram.png"
                )
            )
        else:
            self.amplitudeTensor = amplitudeTensor

        if phaseTensor is None:
            self.phaseTensor = utilities.phase_tensor_generator(
                ".\data\images\sample_hologram.png"
            )
        else:
            self.phaseTensor = phaseTensor

        if not isinstance(self.amplitudeTensor, torch.Tensor) or not isinstance(
            self.phaseTensor, torch.Tensor
        ):
            raise ValueError("Amplitude tensor or phase tensor is required")
        if self.amplitudeTensor.shape != self.phaseTensor.shape:
            raise ValueError("Amplitude and phase tensors must have the same shape")
        
        self.sourcePlain = utilities.complex_plain(self.amplitudeTensor, self.phaseTensor)

        self.distances = distances
        self.pixel_pitch = pixel_pitch
        self.wave_length = wave_length

        self.samplingRowNum = self.amplitudeTensor.shape[-2]
        self.samplingColNum = self.amplitudeTensor.shape[-1]

        self.padding = padding
        self.debug = debug

        if device == "cuda":
            device = utilities.try_gpu()
        self.device = device

    def frequencyMesh(self):
        '''
        Generate the 3-D mesh for the spatial frequency in the z-direction(w)

        Returns:
        w_mesh: 3-D tensor, the mesh for the spatial frequency in the z-direction(w)
        '''
        self.freq_x = torch.fft.fftfreq(self.samplingRowNum, self.pixel_pitch)
        self.freq_y = torch.fft.fftfreq(self.samplingColNum, self.pixel_pitch)

        mesh_x_y = self.freq_x.unsqueeze(1) ** 2 + self.freq_y.unsqueeze(0) ** 2
        mesh_wave_length = 1 / self.wave_length**2
        w_mesh = torch.sqrt(
            torch.clamp(
                mesh_wave_length.unsqueeze(1).unsqueeze(2) - mesh_x_y.unsqueeze(0),
                min=0,
            )
        )

        if self.debug:
            print(
                "The highest positive frequency is u = {} and v = {}.".format(
                    self.freq_x.max(), self.freq_y.max()
                )
            )
            print(
                "The highest negative frequency is u = {} and v = {}.".format(
                    self.freq_x.min(), self.freq_y.min()
                )
            )
            print(
                "The resolution of the frequency is u = {} and v = {}.".format(
                    self.freq_x[1] - self.freq_x[0], self.freq_y[1] - self.freq_y[0]
                )
            )

        return w_mesh

    def band_limited_mask(self):

        S_height = (
            self.samplingRowNum * self.pixel_pitch
        )  # the height of the sampling plain
        S_width = (
            self.samplingColNum * self.pixel_pitch
        )  # the width of the sampling plain

        d_u = 1 / S_height
        d_v = 1 / S_width

        # wave_length = self.wave_length.unsqueeze(0)
        # distances = self.distances.unsqueeze(1)

        u_limit = 1 / (torch.sqrt((2 * d_u * (self.distances.unsqueeze(1))) ** 2 + 1) * (self.wave_length.unsqueeze(0)))
        v_limit = 1 / (torch.sqrt((2 * d_v * (self.distances.unsqueeze(1))) ** 2 + 1) * (self.wave_length.unsqueeze(0)))

        mask_u = torch.abs(self.freq_x).unsqueeze(0).unsqueeze(1).unsqueeze(
            3
        ) < u_limit.unsqueeze(2).unsqueeze(3)
        mask_v = torch.abs(self.freq_y).unsqueeze(0).unsqueeze(1).unsqueeze(
            2
        ) < v_limit.unsqueeze(2).unsqueeze(3)

        mask = mask_u & mask_v

        if self.debug:
            print(
                "The maximum frequency is clipped to u = {} and v = {}.".format(
                    u_limit, v_limit
                )
            )

        return mask

    def band_limited_angular_spectrum_multichannels(
        self,
        band_limit=True,
    ):
        w = self.frequencyMesh()

        # transfer function
        H_FR = torch.exp(2j * math.pi * self.distances.unsqueeze(1).unsqueeze(2).unsqueeze(3) * w)
        # H_FR = torch.exp(1j * math.pi * z * (2/wave_length.unsqueeze(1).unsqueeze(2)-wave_length.unsqueeze(1).unsqueeze(2)*freq_cube))

        if band_limit:
            mask = self.band_limited_mask()
            print(mask.shape)
            H_FR = H_FR * mask

        G_0 = torch.fft.fft2(self.sourcePlain)
        G_z = G_0 * H_FR

        # inverse fourier transform
        g_z_complex = torch.fft.ifft2(G_z)

        return g_z_complex
