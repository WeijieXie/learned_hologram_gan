import torch
from . import utilities


class bandLimitedAngularSpectrumMethod:
    """
    The band limited angular spectrum method for the hologram reconstruction.
    batch/distances * channels * rows * cols
    It is worth noting that:
    1. the band_limit flag is not implemented yet.
    2. it DOES NOT support batch processing and multi-distance processing at the same time.
    In test_bandlimited_agular_spectrum_approach, dim = 0 is interpreted as the different distances.
    While in network, dim = 0 is interpreted as the batch size.

    Attributes:
        samplingRowNum (int): The number of rows in the hologram.
        samplingColNum (int): The number of columns in the hologram.
        pixel_pitch (float): The pixel pitch of the hologram.
        wave_length (torch.Tensor): The wave length of the light.
        band_limit (bool): The flag of band limiting.
        device (torch.device): The device of the calculation.
        freq_x (torch.Tensor): The frequency grid in x direction.
        freq_y (torch.Tensor): The frequency grid in y direction.
        diffraction_limited_mask (torch.Tensor): The diffraction limited mask.
        w_grid (torch.Tensor): The grid of w values.
    """

    def __init__(
        self,
        sample_row_num=192,
        sample_col_num=192,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
    ):
        self.samplingRowNum = sample_row_num
        self.samplingColNum = sample_col_num
        self.pixel_pitch = pixel_pitch
        self.wave_length = wave_length
        self.band_limit = band_limit
        self.device = utilities.try_gpu() if cuda else torch.device("cpu")

        self.freq_x = torch.fft.fftfreq(self.samplingRowNum, self.pixel_pitch)
        self.freq_y = torch.fft.fftfreq(self.samplingColNum, self.pixel_pitch)

        self.diffraction_limited_mask = self.generate_diffraction_limited_mask().to(
            self.device
        )
        self.w_grid = self.generate_w_grid().to(self.device)

        if band_limit:
            pass

    def __call__(
        self,
        amplitute_tensor,
        phase_tensor,
        distances,
        spacial_frequency_filter=None,
    ):
        """
        The forward function of the angular spectrum method.
        Working on the cpu/gpu.

        Args:
            amplitute_tensor (torch.Tensor): The amplitute tensor of the hologram.
            phase_tensor (torch.Tensor): The phase tensor of the hologram.
            distances (torch.Tensor): The distances of the diffractions.
            spacial_frequency_filter (torch.Tensor): The spacial frequency filter.

        Returns:
            torch.Tensor: The intensity of the hologram.
        """
        G_0 = torch.fft.fft2(amplitute_tensor * torch.exp(1j * phase_tensor))
        H = self.generate_transfer_function(distances)
        G_z = G_0 * H * self.diffraction_limited_mask
        intensity = torch.abs(torch.fft.ifft2(G_z)) ** 2
        return intensity

    def propagate_AP2AP(
        self,
        amp_phs_tensor_0,
        distances,
    ):
        """
        Propagate the amplitude and phase tensor of the object to the amplitude and phase tensor of the image.

        Args:
            amp_phs_tensor_0 (torch.Tensor): The amplitude and phase tensor of the object whose shape is (batch_size, 6, samplingRowNum, samplingColNum).
            distances (torch.Tensor): The distances of the diffractions.

        Returns:
            torch.Tensor: The amplitude and phase tensor of the image whose shape is (batch_size, 6, samplingRowNum, samplingColNum).
        """
        G_0 = torch.fft.fft2(
            amp_phs_tensor_0.view(-1, 3, 2, self.samplingRowNum, self.samplingColNum)[
                :, :, 0
            ]
            * torch.exp(
                1j
                * amp_phs_tensor_0.view(
                    -1, 3, 2, self.samplingRowNum, self.samplingColNum
                )[:, :, 1]
            )
        )
        H = self.generate_transfer_function(distances)
        g_z = torch.fft.ifft2(G_0 * H) # NOTICE: if the direction of the propagation is backward, need to divide H
        return torch.cat((torch.abs(g_z), torch.angle(g_z)), dim=1)

    def propagate_P2I(
        self,
        phase_tensor,
        distances,
    ):
        G_0 = torch.fft.fft2(torch.exp(1j * phase_tensor))
        H = self.generate_transfer_function(distances)
        G_z = G_0 * H * self.diffraction_limited_mask
        return torch.abs(torch.fft.ifft2(G_z)) ** 2

    def generate_diffraction_limited_mask(self):
        """
        Generate a diffraction limited mask for the angular spectrum method to simulate the imaging system.
        Working on the cpu.

        Returns:
            torch.Tensor: The diffraction limited mask.
        """
        return utilities.generate_custom_frequency_mask(
            sample_row_num=self.samplingRowNum,
            sample_col_num=self.samplingColNum,
            x=self.samplingRowNum // 3,
            y=self.samplingRowNum // 3 * self.samplingColNum // self.samplingRowNum,
        )

    def generate_w_grid(self):
        """
        Generate a grid of w values for the angular spectrum method.
        Working on the cpu.

        Returns:
            torch.Tensor: The grid of w values.
        """
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
        """
        Generate the transfer function for the angular spectrum method.
        Working on the gpu.

        Args:
            distances (torch.Tensor): The distances of the diffractions.

        Returns:
            torch.Tensor: The transfer function.
        """
        H = torch.exp(
            -2j
            * torch.pi
            * distances.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            * self.w_grid
        )

        return H.squeeze()


class bandLimitedAngularSpectrumMethod_for_single_fixed_distance:
    """
    The band limited angular spectrum method for the hologram reconstruction.
    This version is designed for the case that the distance is fixed.

    Attributes:
        samplingRowNum (int): The number of rows in the hologram.
        samplingColNum (int): The number of columns in the hologram.
        pixel_pitch (float): The pixel pitch of the hologram.
        wave_length (torch.Tensor): The wave length of the light.
        band_limit (bool): The flag of band limiting.
        device (torch.device): The device of the calculation.
        freq_x (torch.Tensor): The frequency grid in x direction.
        freq_y (torch.Tensor): The frequency grid in y direction.
        diffraction_limited_mask (torch.Tensor): The diffraction limited mask.
        distance (float): The distance of the diffraction.
        band_limited_mask (torch.Tensor): The band limited mask.
        H (torch.Tensor): The transfer function.

    """

    def __init__(
        self,
        sample_row_num=192,
        sample_col_num=192,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
        distance=2.5e-2,
    ):
        self.samplingRowNum = sample_row_num
        self.samplingColNum = sample_col_num
        self.pixel_pitch = pixel_pitch
        self.wave_length = wave_length
        self.band_limit = band_limit
        self.device = utilities.try_gpu() if cuda else torch.device("cpu")

        self.freq_x = torch.fft.fftfreq(self.samplingRowNum, self.pixel_pitch)
        self.freq_y = torch.fft.fftfreq(self.samplingColNum, self.pixel_pitch)

        self.diffraction_limited_mask = self.generate_diffraction_limited_mask().to(
            self.device
        )

        if band_limit:
            pass

        # these parameters are fixed to accelerate the calculation in network
        self.distance = distance
        # self.band_limited_mask = self.generate_band_limited_mask().to(self.device)
        self.H = self.generate_transfer_function().to(self.device)

    def __call__(
        self,
        amplitute_tensor,
        phase_tensor,
    ):
        G_0 = torch.fft.fft2(amplitute_tensor * torch.exp(1j * phase_tensor))

        # G_z = G_0 * self.H * self.diffraction_limited_mask * self.band_limited_mask
        G_z = G_0 * self.H * self.diffraction_limited_mask

        intensity = torch.abs(torch.fft.ifft2(G_z)) ** 2
        return intensity

    def propagate_AP2AP(
        self,
        amp_phs_tensor_0,
    ):
        """
        Propagate the amplitude and phase tensor of the object to the amplitude and phase tensor of the image.
        The propagation direction is backward.
        
        Args:
            amp_phs_tensor_0 (torch.Tensor): The amplitude and phase tensor of the object whose shape is (batch_size, 6, samplingRowNum, samplingColNum).

        Returns:
            torch.Tensor: The amplitude and phase tensor of the image whose shape is (batch_size, 6, samplingRowNum, samplingColNum).
        """
        G_0 = torch.fft.fft2(
            amp_phs_tensor_0.view(-1, 3, 2, self.samplingRowNum, self.samplingColNum)[
                :, :, 0
            ]
            * torch.exp(
                1j
                * amp_phs_tensor_0.view(
                    -1, 3, 2, self.samplingRowNum, self.samplingColNum
                )[:, :, 1]
            )
        )
        g_z = torch.fft.ifft2(G_0 / self.H) # because of the direction of the propagation
        return torch.cat((torch.abs(g_z), torch.angle(g_z)), dim=1)

    def propagate_P2I(
        self,
        phase_tensor,
    ):
        """
        Propagate the phase tensor of the object to the intensity tensor of the image.
        The propagation direction is forward.

        Args:
            phase_tensor (torch.Tensor): The phase tensor of the object.

        Returns:
            torch.Tensor: The intensity tensor of the image.
        """
        G_0 = torch.fft.fft2(torch.exp(1j * phase_tensor))
        G_z = G_0 * self.H * self.diffraction_limited_mask
        return torch.abs(torch.fft.ifft2(G_z)) ** 2

    def generate_diffraction_limited_mask(self):
        """
        Generate a diffraction limited mask for the angular spectrum method to simulate the imaging system.
        Working on the cpu.

        Returns:
            torch.Tensor: The diffraction limited mask.
        """
        return utilities.generate_custom_frequency_mask(
            sample_row_num=self.samplingRowNum,
            sample_col_num=self.samplingColNum,
            x=self.samplingRowNum // 3,
            y=self.samplingRowNum // 3 * self.samplingColNum // self.samplingRowNum,
        )

    def generate_w_grid(self):
        """
        Generate a grid of w values for the angular spectrum method.
        Working on the cpu.

        Returns:
            torch.Tensor: The grid of w values.
        """
        squared_u_v_grid = self.freq_x.unsqueeze(1) ** 2 + self.freq_y.unsqueeze(0) ** 2
        w_grid = torch.sqrt(
            torch.clamp(
                (1 / self.wave_length**2).unsqueeze(1).unsqueeze(2)
                - squared_u_v_grid.unsqueeze(0),
                min=0,
            )
        )
        return w_grid

    def generate_band_limited_mask(self):
        d_x_0 = 1 / (self.samplingRowNum * self.pixel_pitch)
        d_y_0 = 1 / (self.samplingColNum * self.pixel_pitch)

        u_limit = 1 / (
            torch.sqrt((2 * d_x_0 * (self.distance.unsqueeze(1))) ** 2 + 1)
            * (self.wave_length.unsqueeze(0))
        )
        v_limit = 1 / (
            torch.sqrt((2 * d_y_0 * (self.distance.unsqueeze(1))) ** 2 + 1)
            * (self.wave_length.unsqueeze(0))
        )

        mask_u = torch.abs(self.freq_x).unsqueeze(0).unsqueeze(1).unsqueeze(
            3
        ) < u_limit.unsqueeze(2).unsqueeze(3)
        mask_v = torch.abs(self.freq_y).unsqueeze(0).unsqueeze(1).unsqueeze(
            2
        ) < v_limit.unsqueeze(2).unsqueeze(3)

        return mask_u & mask_v

    def generate_transfer_function(self):
        H = torch.exp(-2j * torch.pi * self.distance * self.generate_w_grid())
        return H
