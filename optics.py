import torch
import utilities


class bandLimitedAngularSpectrumMethod:

    def __init__(
        self,
        amplitudeTensor=None,
        phaseTensor=None,
        distances=torch.Tensor([0.0]),
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=True,
        fresnel_approximation=False,
        padding=False,
        debug=False,
        device="cuda",
    ):
        if device == "cuda":
            device = utilities.try_gpu()
        self.device = device

        if phaseTensor is None:
            self.phaseTensor = utilities.phase_tensor_generator(
                ".\data\images\sample_hologram.png"
            ).to(self.device)
        else:
            self.phaseTensor = phaseTensor.to(self.device)

        if amplitudeTensor is None:
            self.amplitudeTensor = (
                utilities.amplitude_tensor_generator_for_phase_only_hologram(
                    self.phaseTensor
                ).to(self.device)
            )
        else:
            self.amplitudeTensor = amplitudeTensor.to(self.device)

        if self.amplitudeTensor.shape != self.phaseTensor.shape:
            raise ValueError("Amplitude and phase tensors must have the same shape")

        self.sourcePlain = utilities.complex_plain(
            self.amplitudeTensor, self.phaseTensor
        )

        self.distances = distances
        self.pixel_pitch = pixel_pitch
        self.wave_length = wave_length

        self.samplingRowNum = self.amplitudeTensor.shape[-2]
        self.samplingColNum = self.amplitudeTensor.shape[-1]

        self.band_limit = band_limit
        self.fresnel_approximation = fresnel_approximation
        self.padding = padding
        self.debug = debug

        self.freq_x = torch.fft.fftfreq(self.samplingRowNum, self.pixel_pitch)
        self.freq_y = torch.fft.fftfreq(self.samplingColNum, self.pixel_pitch)
        self.mesh_x_y = self.freq_x.unsqueeze(1) ** 2 + self.freq_y.unsqueeze(0) ** 2

        self.w_mesh = self.frequencyMesh()
        self.bandLimitedMask = self.band_limited_mask().to(self.device)
        self.H = self.transfer_function(self.fresnel_approximation).to(self.device)

    def __del__(self):
        del self.amplitudeTensor
        del self.phaseTensor
        del self.sourcePlain
        del self.distances
        del self.pixel_pitch
        del self.wave_length
        del self.samplingRowNum
        del self.samplingColNum
        del self.band_limit
        del self.fresnel_approximation
        del self.padding
        del self.debug
        del self.freq_x
        del self.freq_y
        del self.mesh_x_y
        del self.w_mesh
        del self.bandLimitedMask
        del self.H

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def frequencyMesh(self):

        mesh_wave_length = 1 / self.wave_length**2

        # evanescent wave
        w_mesh = torch.sqrt(
            torch.clamp(
                mesh_wave_length.unsqueeze(1).unsqueeze(2) - self.mesh_x_y.unsqueeze(0),
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

        if self.band_limit:
            S_height = (
                self.samplingRowNum * self.pixel_pitch
            )  # the height of the sampling plain
            S_width = (
                self.samplingColNum * self.pixel_pitch
            )  # the width of the sampling plain

            d_u = 1 / S_height
            d_v = 1 / S_width

            u_limit = 1 / (
                torch.sqrt((2 * d_u * (self.distances.unsqueeze(1))) ** 2 + 1)
                * (self.wave_length.unsqueeze(0))
            )
            v_limit = 1 / (
                torch.sqrt((2 * d_v * (self.distances.unsqueeze(1))) ** 2 + 1)
                * (self.wave_length.unsqueeze(0))
            )

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
        else:
            mask = utilities.mask_generator(
                self.samplingRowNum, self.samplingColNum, 30, 50
            )

        return mask

    def transfer_function(self, FresnelApproximation=False):
        if FresnelApproximation:
            # Fresnel's approximation
            H = torch.exp(
                1j
                * torch.pi
                * self.distances.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                * (
                    2 / self.wave_length.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    - self.wave_length.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    * self.mesh_x_y.unsqueeze(0).unsqueeze(1)
                )
            )
            print("using Fresnel's approximation")
            print(H.shape)
        else:
            # transfer function
            H = torch.exp(
                2j
                * torch.pi
                * self.distances.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                * self.w_mesh
            )
            print("using Angular Spectrum Method")
            print(H.shape)
        return H

    def band_limited_angular_spectrum_multichannels(
        self,
    ):

        G_0 = torch.fft.fft2(self.sourcePlain).to(self.device)
        G_z = G_0 * self.H * self.bandLimitedMask
        g_z_complex = torch.fft.ifft2(G_z)

        return g_z_complex
