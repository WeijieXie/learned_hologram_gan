import torch
from . import utilities


class bandLimitedAngularSpectrumMethod:

    def __init__(
        self,
        sample_row_num=192,
        sample_col_num=192,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=True,
        padding=False,
        cuda=False,
    ):
        self.samplingRowNum = sample_row_num
        self.samplingColNum = sample_col_num
        self.pixel_pitch = pixel_pitch
        self.wave_length = wave_length
        self.band_limit = band_limit
        self.padding = padding
        self.device = utilities.try_gpu() if cuda else torch.device("cpu")

        self.freq_x = torch.fft.fftfreq(self.samplingRowNum, self.pixel_pitch)
        self.freq_y = torch.fft.fftfreq(self.samplingColNum, self.pixel_pitch)

        self.w_grid = self.generate_w_grid()

        if padding:
            pass

        if band_limit:
            pass

    def __call__(
        self,
        amplitute_tensor,
        phase_tensor,
        distances,
        spacial_frequency_filter,
    ):
        G_0 = torch.fft.fft2(amplitute_tensor * torch.exp(1j * phase_tensor))
        H = self.generate_transfer_function(distances)
        G_z = G_0 * H * spacial_frequency_filter
        intensity = torch.abs(torch.fft.ifft2(G_z)) ** 2
        # intensity = utilities.intensity_calculator(torch.fft.ifft2(G_z))
        return intensity

    def generate_custom_mask(self):
        pass

    def generate_w_grid(self):
        squared_u_v_grid = self.freq_x.unsqueeze(1) ** 2 + self.freq_y.unsqueeze(0) ** 2
        w_grid = torch.sqrt(
            torch.clamp(
                (1 / self.wave_length**2).unsqueeze(1).unsqueeze(2)
                - squared_u_v_grid.unsqueeze(0),
                min=0,
            )
        )
        return w_grid

    def generate_band_limited_mask(self, distances):
        d_x_0 = 1 / (self.samplingRowNum * self.pixel_pitch)
        d_y_0 = 1 / (self.samplingColNum * self.pixel_pitch)

        u_limit = 1 / (
            torch.sqrt((2 * d_x_0 * (distances.unsqueeze(1))) ** 2 + 1)
            * (self.wave_length.unsqueeze(0))
        )
        v_limit = 1 / (
            torch.sqrt((2 * d_y_0 * (distances.unsqueeze(1))) ** 2 + 1)
            * (self.wave_length.unsqueeze(0))
        )

        mask_u = torch.abs(self.freq_x).unsqueeze(0).unsqueeze(1).unsqueeze(
            3
        ) < u_limit.unsqueeze(2).unsqueeze(3)
        mask_v = torch.abs(self.freq_y).unsqueeze(0).unsqueeze(1).unsqueeze(
            2
        ) < v_limit.unsqueeze(2).unsqueeze(3)

        return mask_u & mask_v

    def generate_transfer_function(self, distances):
        H = torch.exp(
            -2j
            * torch.pi
            * distances.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            * self.w_grid
        )

        return H
