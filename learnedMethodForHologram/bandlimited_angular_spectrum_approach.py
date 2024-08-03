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
        pad_size=0,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
    ):

        self.originalRowNum = sample_row_num
        self.originalColNum = sample_col_num
        self.samplingRowNum = sample_row_num + 2 * pad_size
        self.samplingColNum = sample_col_num + 2 * pad_size
        self.pad_size = pad_size
        self.pixel_pitch = pixel_pitch
        self.wave_length = wave_length
        self.band_limit = band_limit
        self.device = utilities.try_gpu() if cuda else torch.device("cpu")

        self.freq_x = torch.fft.fftfreq(self.samplingRowNum, self.pixel_pitch)
        self.freq_y = torch.fft.fftfreq(self.samplingColNum, self.pixel_pitch)

        # simulate the imaging system
        self.diffraction_limited_mask = self.generate_diffraction_limited_mask(0.5).to(
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
        G_0 = torch.fft.fft2(
            self.padding(amplitute_tensor * torch.exp(1j * phase_tensor))
        )
        H = self.generate_transfer_function(distances)
        G_z = G_0 * H * self.diffraction_limited_mask
        # G_z = G_0 * H
        # intensity = torch.abs(self.cropping(torch.fft.ifft2(G_z))) ** 2
        intensity = torch.abs(self.cropping(torch.fft.ifft2(G_z)))

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
            self.padding(
                amp_phs_tensor_0.view(
                    -1, 3, 2, self.samplingRowNum, self.samplingColNum
                )[:, :, 0]
                * torch.exp(
                    1j
                    * amp_phs_tensor_0.view(
                        -1, 3, 2, self.samplingRowNum, self.samplingColNum
                    )[:, :, 1]
                )
            )
        )
        H = self.generate_transfer_function(distances)

        # NOTICE: if the direction of the propagation is backward, need to divide H!!!!!!!!
        g_z = self.cropping(torch.fft.ifft2(G_0 * H))

        return torch.cat((torch.abs(g_z), torch.angle(g_z)), dim=1)

    def propagate_P2I(
        self,
        phase_tensor,
        distances,
    ):
        G_0 = torch.fft.fft2(self.padding(torch.exp(1j * phase_tensor)))
        H = self.generate_transfer_function(distances)
        G_z = G_0 * H * self.diffraction_limited_mask
        return torch.abs(self.cropping(torch.fft.ifft2(G_z))) ** 2

    def generate_diffraction_limited_mask(self, radius_coefficient):
        """
        Generate a diffraction limited mask for the angular spectrum method to simulate the imaging system.
        Working on the cpu.

        Returns:
            torch.Tensor: The diffraction limited mask.
        """
        return utilities.generate_circular_frequency_mask(
            sample_row_num=self.samplingRowNum,
            sample_col_num=self.samplingColNum,
            radius=min(self.samplingRowNum, self.samplingColNum) * radius_coefficient,
            # radius=min(self.samplingRowNum, self.samplingColNum) // 2,
            # which picks 2/3 frequencies on the frequency domain
            # radius=192,
        ).to(self.device)

    def generate_w_grid(self):
        """
        Generate a grid of w values for the angular spectrum method.
        Working on the cpu.

        Returns:
            torch.Tensor: The grid of w values, which is a 3D tensor.
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

        return H

    def padding(self, tensor):
        """
        Padding the tensor with zeros.

        Args:
            tensor (torch.Tensor): The tensor to be padded.
            pad_size (int): The size of the padding.

        Returns:
            torch.Tensor: The padded tensor.
        """
        if self.pad_size == 0:
            return tensor
        else:
            return torch.nn.functional.pad(
                tensor,
                (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                mode="constant",
                value=0,
            )

    def cropping(self, tensor):
        """
        Invert the padding operation.

        Args:
            tensor (torch.Tensor): The tensor to be inverted.
            pad_size (int): The size of the padding.

        Returns:
            torch.Tensor: The inverted tensor.
        """
        if self.pad_size == 0:
            return tensor
        else:
            return tensor[
                :, :, self.pad_size : -self.pad_size, self.pad_size : -self.pad_size
            ]


class bandLimitedAngularSpectrumMethod_for_single_fixed_distance(
    bandLimitedAngularSpectrumMethod
):
    """
    The band limited angular spectrum method for the hologram reconstruction.
    This version is designed for the case that the distance is fixed.

    Notice: It CANNOT handle 3D tensor as input.

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
        pad_size=0,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
        distance=torch.tensor([1e-3]),
    ):
        super(
            bandLimitedAngularSpectrumMethod_for_single_fixed_distance, self
        ).__init__(
            sample_row_num,
            sample_col_num,
            pad_size,
            pixel_pitch,
            wave_length,
            band_limit,
            cuda,
        )

        # these parameters are fixed to accelerate the calculation in network
        self.distance = distance
        self.circular_frequency_mask_differentiable_grid = (
            utilities.prepare_circular_frequency_mask_differentiable_grid(
                self.samplingRowNum, self.samplingColNum
            ).to(self.device)
        )
        self.band_limited_mask = self.generate_band_limited_mask().to(self.device)
        self.H = self.generate_transfer_function().to(self.device)

    def __call__(
        self,
        amplitute_tensor,
        phase_tensor,
    ):
        G_0 = torch.fft.fft2(
            self.padding(amplitute_tensor * torch.exp(1j * phase_tensor))
        )

        # G_z = G_0 * self.H * self.diffraction_limited_mask * self.band_limited_mask
        G_z = G_0 * self.H * self.diffraction_limited_mask

        intensity = torch.abs(self.cropping(torch.fft.ifft2(G_z)))
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
            self.padding(
                amp_phs_tensor_0.view(
                    -1, 3, 2, self.samplingRowNum, self.samplingColNum
                )[:, :, 0]
                * torch.exp(
                    1j
                    * amp_phs_tensor_0.view(
                        -1, 3, 2, self.samplingRowNum, self.samplingColNum
                    )[:, :, 1]
                )
            )
        )
        g_z = self.cropping(
            torch.fft.ifft2(G_0 / self.H)
        )  # because of the direction of the propagation
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
        G_0 = torch.fft.fft2(self.padding(torch.exp(1j * phase_tensor)))
        G_z = G_0 * self.H * self.diffraction_limited_mask
        return torch.abs(self.cropping(torch.fft.ifft2(G_z))) ** 2

    def propagate_P2IP(
        self,
        phase_tensor,
    ):
        G_0 = torch.fft.fft2(self.padding(torch.exp(1j * phase_tensor)))
        G_z = G_0 * self.H * self.diffraction_limited_mask
        g_z = self.cropping(torch.fft.ifft2(G_z))
        return torch.cat(
            (torch.abs(g_z) ** 2, torch.angle(g_z)), dim=1
        )  # dim = 0 is the batch size

    def propagate_P2AP(
        self,
        phase_tensor,
    ):
        G_0 = torch.fft.fft2(self.padding(torch.exp(1j * phase_tensor)))
        G_z = G_0 * self.H * self.diffraction_limited_mask
        g_z = self.cropping(torch.fft.ifft2(G_z))
        return torch.cat(
            (torch.abs(g_z), torch.angle(g_z)), dim=1
        )  # dim = 0 is the batch size

    def propagate_P2AAP(
        self,
        phase_tensor,
    ):
        """
        This method is designed for v4. It returns a tensor with 9 channels.
        """
        G_0 = torch.fft.fft2(self.padding(torch.exp(1j * phase_tensor)))
        G_z = G_0 * self.H
        G_z_4f = G_z * self.diffraction_limited_mask
        g_z_6_channels = self.cropping(torch.fft.ifft2(torch.cat((G_z_4f, G_z), dim=1)))

        # return 3 + 3 + 3 : 3 channels of filtered amplitude + 3 channels of amplitude + 3 channels of phs
        return torch.cat(
            (torch.abs(g_z_6_channels), torch.angle(g_z_6_channels[:, 3:])), dim=1
        )

    # --------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------For GAN--------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------

    def propagate_AP2AP_backward(
        self,
        amp_z,
        phs_z,
    ):
        """
        For GAN
        """
        G_z = torch.fft.fft2(self.padding(amp_z * torch.exp(1j * phs_z)))
        g_0 = self.cropping(torch.fft.ifft2(G_z / self.H))
        return torch.abs(g_0), torch.angle(g_0)
    
    def propagate_AP2C_backward(
        self,
        amp_z,
        phs_z,
    ):
        """
        For GAN
        """
        G_z = torch.fft.fft2(self.padding(amp_z * torch.exp(1j * phs_z)))
        g_0 = self.cropping(torch.fft.ifft2(G_z / self.H))
        return g_0

    def propagate_POH2AP_forward(
        self,
        phs_0,
        filter_radius_coefficient=torch.tensor(0.5),
    ):
        """
        For GAN
        """
        G_0 = torch.fft.fft2(self.padding(torch.exp(1j * phs_0)))
        G_z_filtered = (
            G_0
            * self.H
            * self.generate_circular_frequency_mask_differentiable(
                filter_radius_coefficient
            )
        )
        spectrum_mean_loss = torch.mean(torch.abs(G_0) - torch.abs(G_z_filtered))
        g_z = self.cropping(torch.fft.ifft2(G_z_filtered))
        return torch.abs(g_z), torch.angle(g_z), spectrum_mean_loss

    def propagate_AP2AP_forward(
        self,
        amp_0,
        phs_0,
        filter_radius_coefficient=torch.tensor(0.5),
    ):
        """
        For GAN
        """
        G_0 = torch.fft.fft2(self.padding(amp_0 * torch.exp(1j * phs_0)))

        # because of the direction of the propagation
        g_z = self.cropping(
            torch.fft.ifft2(
                G_0
                * self.H
                * self.generate_circular_frequency_mask_differentiable(
                    filter_radius_coefficient
                )
            )
        )
        return torch.abs(g_z), torch.angle(g_z)

    def generate_circular_frequency_mask_differentiable(
        self,
        filter_radius_coefficient,
    ):
        shorter_edge = min(self.samplingRowNum, self.samplingColNum)
        radius = shorter_edge * filter_radius_coefficient
        mask = torch.sigmoid(
            1.0 * (radius - self.circular_frequency_mask_differentiable_grid)
        )

        return mask

    # --------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------For GAN--------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------

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
        H = torch.exp(-2j * torch.pi * self.distance * super().generate_w_grid())
        return H


class bandLimitedAngularSpectrumMethod_for_multiple_distances(
    bandLimitedAngularSpectrumMethod
):
    """
    This version supports multiple distances and batch processing at the same time.
    [a,b,c,d]: with a = batch_size, b = distances_num * 3, c = sample_row_num, d = sample_col_num
    """

    def __init__(
        self,
        sample_row_num=192,
        sample_col_num=192,
        pad_size=192,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
    ):
        super(bandLimitedAngularSpectrumMethod_for_multiple_distances, self).__init__(
            sample_row_num,
            sample_col_num,
            pad_size,
            pixel_pitch,
            wave_length,
            band_limit,
            cuda,
        )

    def __call__(
        self,
        amplitute_tensor,
        phase_tensor,
        distances,
    ):
        distances = distances.to(self.device)
        distances_num = distances.shape[0]
        G_0 = torch.fft.fft2(
            self.padding(amplitute_tensor * torch.exp(1j * phase_tensor))
        )
        H = self.generate_transfer_function(distances) * self.diffraction_limited_mask
        # H = self.generate_transfer_function(distances)

        G_z = (G_0.unsqueeze(1) * H).view(
            -1, distances_num, 3, self.samplingRowNum, self.samplingColNum
        )
        intensity = torch.abs(self.cropping(torch.fft.ifft2(G_z)))
        return intensity

    def cropping(self, tensor):
        """
        Invert the padding operation.

        Args:
            tensor (torch.Tensor): The tensor to be inverted.
            pad_size (int): The size of the padding.

        Returns:
            torch.Tensor: The inverted tensor.
        """
        if self.pad_size == 0:
            return tensor
        else:
            return tensor[
                :, :, :, self.pad_size : -self.pad_size, self.pad_size : -self.pad_size
            ]
