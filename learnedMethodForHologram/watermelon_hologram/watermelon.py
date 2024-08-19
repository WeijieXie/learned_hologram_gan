import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.autograd as autograd

from .generator import Generator
from .discriminator import WGANGPDiscriminator192, fakeDiscriminator
from .loss_func import (
    perceptualLoss,
    total_variation_loss,
    focal_sincos_phase_gradient_loss,
    phase_sincos_gradient_loss,
    focal_sincos_phase_loss,
    plain_phase_loss,
)

from ..neural_network_components import fakeChannelWiseSymmetricConv

from ..angular_spectrum_method import (
    bandLimitedAngularSpectrumMethod_for_multiple_distances,
)

from ..utilities import try_gpu, multi_sample_plotter, tensor_normalizor_2D

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)

import json


class watermelon:
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        kernel_size=3,
        distance_stack=torch.linspace(-1.5e-4, 0.0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        self.device = try_gpu() if cuda else torch.device("cpu")

        self.distance_stack = distance_stack.to(self.device)
        self.distance_num = distance_stack.size(0)

        self.generator = Generator(
            sample_row_num=input_shape[-2],
            sample_col_num=input_shape[-1],
            pad_size=pad_size,
            filter_radius_coefficient=filter_radius_coefficient,
            kernel_size=kernel_size,
            pixel_pitch=3.74e-6,
            wave_length=torch.tensor([638e-9, 520e-9, 450e-9]),
            distance=torch.tensor([1e-3]),
            pretrained_model_path=pretrained_model_path_G,
        )

        self.discriminator = WGANGPDiscriminator192(
            pretrained_model_path=pretrained_model_path_D,
            cuda=True,
        )

        self.perceptual_loss = perceptualLoss(
            feature_map_layers=[3, 8, 13, 22, 31],
            cuda=True,
        )

        self.propagator = bandLimitedAngularSpectrumMethod_for_multiple_distances(
            sample_row_num=input_shape[-2],
            sample_col_num=input_shape[-1],
            distances=distance_stack,
            pad_size=pad_size,
            filter_radius_coefficient=filter_radius_coefficient,
            pixel_pitch=3.74e-6,
            wave_length=torch.tensor([638e-9, 520e-9, 450e-9]),
            band_limit=False,
            cuda=True,
        )

        if pretrained_model_path_G is not None:
            self.generator.load_state_dict(torch.load(pretrained_model_path_G))
            print(f"Generator loaded from {pretrained_model_path_G}")

        if pretrained_model_path_D is not None:
            self.discriminator.load_state_dict(torch.load(pretrained_model_path_D))
            print(f"Discriminator loaded from {pretrained_model_path_D}")

    def train(
        self,
        data_loader_train,
        data_loader_val,
        phs_gradient_loss_weight=1,
        perceptual_loss_weight=1.0,
        pixel_loss_weight=1.0,
        TV_loss_weight=1e-3,
        discriminator_loss_weight=1.0,
        epoch_num=2,
        lr_G=1e-3,
        lr_D=1e-3,
        save_path_G=None,
        save_path_D=None,
        info_print_interval=100,
        info_plot_interval=600,
        loss_metrics_file=None,
        save_path_img=None,
        checkpoint_iterval=5,
        discriminator_train_ratio=2,
        discriminator_lambda=10,
        step_scheduler_G_gamma=0.1,
        step_scheduler_D_gamma=0.9999,
        visualization_RGBD_AP=None,
    ):

        if save_path_G is None:
            print(
                "!!!!!!The save path of the generator is not specified, the model will not be saved!!!!!!"
            )

        if save_path_D is None:
            print(
                "!!!!!!The save path of the discriminator is not specified, the model will not be saved!!!!!!"
            )

        self.phs_gradient_loss_weight = phs_gradient_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.TV_loss_weight = TV_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight

        self.PSNR_metric = PSNR().to(self.device)
        self.SSIM_metric = SSIM().to(self.device)

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D)

        # step_scheduler_G = ReduceLROnPlateau(
        #     optimizer_G,
        #     "min",
        #     factor=step_scheduler_G_gamma,
        #     patience=5,
        #     verbose=True,
        #     threshold=1e-4,
        #     threshold_mode="rel",
        #     min_lr=1e-6,
        # )

        # step_scheduler_D = ReduceLROnPlateau(
        #     optimizer_D,
        #     "min",
        #     factor=step_scheduler_D_gamma,
        #     patience=100,
        #     verbose=True,
        #     threshold=1e-4,
        #     threshold_mode="rel",
        #     min_lr=1e-6,
        # )

        n_train = 0
        n_batch = 0
        n_batch_last = 0

        with torch.no_grad():

            # create the dictionary to record the losses and metrics
            self.dict_for_losses_metrics = {
                "epoch": [],
                "n_batch_in_epoch": [],
                "n_train": [],
                "n_batch": [],
                "train_losses_tensor": {
                    "focal_phase_gradient_loss": [],
                    "perceptual_loss": [],
                    "pixel_loss": [],
                    "TV_loss": [],
                    "gan_loss": [],
                    "G_loss": [],
                    "D_loss": [],
                },
                "train_metrics_tensor": {"PSNR": [], "SSIM": []},
                "validate_losses_tensor": {
                    "focal_phase_gradient_loss": [],
                    "perceptual_loss": [],
                    "pixel_loss": [],
                    "TV_loss": [],
                    "gan_loss": [],
                    "G_loss": [],
                    "D_loss": [],
                },
                "validate_metrics_tensor": {"PSNR": [], "SSIM": []},
            }

            self.train_losses_tensor = torch.zeros(7).to(self.device)
            self.train_metrics_tensor = torch.zeros(2).to(self.device)

            train_losses_tensor_last = torch.zeros(7).to(self.device)
            train_metrics_tensor_last = torch.zeros(2).to(self.device)

        for epoch in range(epoch_num):

            self.generator.train()
            self.discriminator.train()

            for n_batch_in_epoch, (RGBD, target_amp, target_phs) in enumerate(
                data_loader_train
            ):

                n_batch += 1
                G_batch_size = RGBD.size(0)
                n_train += G_batch_size

                # phase at z = z0
                POH_phs = self.generator(RGBD)

                # filtered frequency of hat at z = z1
                hat_freq = self.generator.part2.propagator.propagate_POH2Freq_forward(
                    POH_phs
                )

                # filtered frequency of target at z = z1
                target_freq = self.propagator.filter_AP2filteredFreq(
                    target_amp, target_phs
                )

                # propagate the hat and target frequency to different distances
                hat_target_freq = torch.cat((hat_freq, target_freq), dim=0)
                hat_target_amp, hat_target_phs = (
                    self.propagator.propagate_multiple_samples_with_random_fixed_multiple_distances_freq2amp(
                        hat_target_freq
                    )
                )

                # the hat and target amplitude at multible distances
                D_batch_size = G_batch_size

                hat_amps = hat_target_amp[:D_batch_size]
                target_amps = hat_target_amp[D_batch_size:]
                hat_phases = hat_target_phs[:D_batch_size]
                target_phases = hat_target_phs[D_batch_size:]

                for _ in range(discriminator_train_ratio):
                    real_validity = self.discriminator(target_amps)
                    fake_validity = self.discriminator(hat_amps.detach())
                    gradient_penalty = self.compute_gradient_penalty(
                        target_amps, hat_amps.detach()
                    )
                    discriminator_loss = (
                        -torch.mean(real_validity) + torch.mean(fake_validity)
                    ) + discriminator_lambda * gradient_penalty

                    optimizer_D.zero_grad()
                    discriminator_loss.backward(retain_graph=True)
                    optimizer_D.step()

                    # record the loss
                    self.train_losses_tensor[-1] += (
                        discriminator_loss.item() / discriminator_train_ratio
                    )

                # calculate the loss from the discriminator
                loss_from_discriminator = -torch.mean(self.discriminator(hat_amps))

                generator_loss = self.G_loss(
                    hat_amps,
                    target_amps,
                    hat_phases,
                    target_phases,
                    loss_from_discriminator,
                    self.train_losses_tensor,
                )

                optimizer_G.zero_grad()
                generator_loss.backward()
                optimizer_G.step()

                with torch.no_grad():

                    # record the metrics
                    self.record_metrics(
                        hat_amps, target_amps, self.train_metrics_tensor
                    )

                    if n_batch % info_print_interval == 0:

                        # validate the generator with the validation dataset
                        validate_losses_tensor_iter, validate_metrics_tensor_iter = (
                            self._validate_generator(data_loader_val)
                        )

                        train_losses_tensor_iter = (
                            self.train_losses_tensor - train_losses_tensor_last
                        ) / (n_batch - n_batch_last)

                        train_metrics_tensor_iter = (
                            self.train_metrics_tensor - train_metrics_tensor_last
                        ) / (n_batch - n_batch_last)

                        print(
                            f"epoch {epoch}, batch {n_batch_in_epoch + 1} ({n_train} samples and {n_batch} batches have been trained):\n"
                            f"      train: focal_phase_gradient_loss {train_losses_tensor_iter[0]}, perceptual_loss {train_losses_tensor_iter[1]}, pixel_loss {train_losses_tensor_iter[2]}, TV_loss {train_losses_tensor_iter[3]}, gan_loss {train_losses_tensor_iter[4]}, G_loss {train_losses_tensor_iter[5]}, D_loss {train_losses_tensor_iter[6]};\n"
                            f"      train: PSNR {train_metrics_tensor_iter[0]}, SSIM {train_metrics_tensor_iter[1]};\n"
                            f"      validate: focal_phase_gradient_loss {validate_losses_tensor_iter[0]}, perceptual_loss {validate_losses_tensor_iter[1]}, pixel_loss {validate_losses_tensor_iter[2]}, TV_loss {validate_losses_tensor_iter[3]}, gan_loss {validate_losses_tensor_iter[4]}, G_loss {validate_losses_tensor_iter[5]}, D_loss {validate_losses_tensor_iter[6]};\n"
                            f"      validate: PSNR {validate_metrics_tensor_iter[0]}, SSIM {validate_metrics_tensor_iter[1]};\n"
                        )

                        self._add_losses_metrics_to_dict(
                            epoch,
                            n_batch_in_epoch,
                            n_train,
                            n_batch,
                            validate_losses_tensor_iter,
                            validate_metrics_tensor_iter,
                            train_losses_tensor_iter,
                            train_metrics_tensor_iter,
                            self.dict_for_losses_metrics,
                        )

                        train_losses_tensor_last = self.train_losses_tensor.clone()
                        train_metrics_tensor_last = self.train_metrics_tensor.clone()
                        n_batch_last = n_batch

                    if n_batch % info_plot_interval == 0:
                        if visualization_RGBD_AP is not None:
                            RGBD, target_amp, target_phs = visualization_RGBD_AP
                            RGBD = RGBD.unsqueeze(0)
                            POH_phs = self.generator(RGBD)
                            amp_hat, phs_hat = (
                                self.generator.part2.propagator.propagate_POH2AP_forward(
                                    POH_phs
                                )
                            )

                            multi_sample_plotter(
                                tensor_normalizor_2D(
                                    torch.cat(
                                        (
                                            amp_hat,
                                            phs_hat,
                                        ),
                                        dim=0,
                                    ),
                                ),
                                titles=[
                                    f"amp_hat in epoch {epoch}, batch {n_batch_in_epoch + 1}",
                                    f"phs_hat in epoch {epoch}, batch {n_batch_in_epoch + 1}",
                                ],
                                rgb_img=True,
                                save_dir=save_path_img,
                            )
                            print(
                                f"visualization saved at epoch {epoch}, batch {n_batch_in_epoch + 1}"
                            )

            # step_scheduler_G.step(self.validate_losses_tensor[4])
            # step_scheduler_D.step(self.validate_losses_tensor[5])

            # checkpoint
            if epoch % checkpoint_iterval == 0:
                if save_path_G is not None:
                    check_point_path = save_path_G.replace(".pth", f"_epoch{epoch}.pth")
                    torch.save(self.generator.state_dict(), check_point_path)
                    print(f"Generator saved to {check_point_path}")

                if save_path_D is not None:
                    check_point_path = save_path_D.replace(".pth", f"_epoch{epoch}.pth")
                    torch.save(self.discriminator.state_dict(), check_point_path)
                    print(f"Discriminator saved to {check_point_path}")

                if loss_metrics_file is not None:
                    self._save_losses_metrics_to_dict(loss_metrics_file)
                    print(f"losses and metrics saved to {loss_metrics_file}")

                with torch.no_grad():
                    if visualization_RGBD_AP is not None:
                        RGBD, target_amp, target_phs = visualization_RGBD_AP
                        RGBD = RGBD.unsqueeze(0)
                        POH_phs = self.generator(RGBD)
                        amp_hat, phs_hat = (
                            self.generator.part2.propagator.propagate_POH2AP_forward(
                                POH_phs
                            )
                        )

                        multi_sample_plotter(
                            tensor_normalizor_2D(
                                torch.cat(
                                    (
                                        amp_hat,
                                        phs_hat,
                                    ),
                                    dim=0,
                                ),
                            ),
                            titles=[
                                f"amp_hat in epoch {epoch}",
                                f"phs_hat in epoch {epoch}",
                            ],
                            rgb_img=True,
                            save_dir=save_path_img,
                        )
                        print(f"visualization saved at epoch {epoch}")

        if save_path_G is not None:
            torch.save(self.generator.state_dict(), save_path_G)
            print(f"Generator saved to {save_path_G}")

        if save_path_D is not None:
            torch.save(self.discriminator.state_dict(), save_path_D)
            print(f"Discriminator saved to {save_path_D}")

        if loss_metrics_file is not None:
            self._save_losses_metrics_to_dict(loss_metrics_file)
            print(f"losses and metrics saved to {loss_metrics_file}")

    def G_loss(
        self,
        hat_amps,
        target_amps,
        hat_phs,
        target_phs,
        loss_from_discriminator,
        recorder=None,
    ):
        phs_loss = (
            focal_sincos_phase_gradient_loss(hat_phs, target_phs)
            * self.phs_gradient_loss_weight
        )
        perceptual_loss = (
            self.perceptual_loss(hat_amps, target_amps) * self.perceptual_loss_weight
        )
        pixel_loss = F.mse_loss(hat_amps, target_amps) * self.pixel_loss_weight
        TV_loss = total_variation_loss(hat_amps, target_amps) * self.TV_loss_weight
        gan_loss = loss_from_discriminator * self.discriminator_loss_weight
        loss = phs_loss + perceptual_loss + pixel_loss + TV_loss + gan_loss

        with torch.no_grad():
            recorder += torch.tensor(
                [phs_loss, perceptual_loss, pixel_loss, TV_loss, gan_loss, loss, 0.0],
                device=self.device,
            )

        return loss

    def record_metrics(
        self,
        hat_amps,
        target_amps,
        recorder=None,
    ):
        with torch.no_grad():
            PSNR_value = self.PSNR_metric(hat_amps, target_amps)
            SSIM_value = self.SSIM_metric(hat_amps, target_amps)
            recorder += torch.tensor([PSNR_value, SSIM_value], device=self.device)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates, requires_grad=False).to(self.device)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _validate_generator(
        self,
        data_loader_val,
    ):

        self.generator.eval()
        self.discriminator.eval()

        validate_losses_tensor = torch.zeros(7).to(self.device)
        validate_metrics_tensor = torch.zeros(2).to(self.device)

        with torch.no_grad():

            n_batch = 0
            for i, (RGBD, target_amp, target_phs) in enumerate(data_loader_val):

                n_batch += 1
                G_batch_size = RGBD.size(0)

                # phase at z = z0
                POH_phs = self.generator(RGBD)

                # filtered frequency of hat at z = z1
                hat_freq = self.generator.part2.propagator.propagate_POH2Freq_forward(
                    POH_phs
                )

                # filtered frequency of target at z = z1
                target_freq = self.propagator.filter_AP2filteredFreq(
                    target_amp, target_phs
                )

                # propagate the hat and target frequency to different distances
                hat_target_freq = torch.cat((hat_freq, target_freq), dim=0)
                hat_target_amp, hat_target_phs = (
                    self.propagator.propagate_multiple_samples_with_all_fixed_multiple_distances_freq2amp(
                        hat_target_freq
                    )
                )

                D_batch_size = G_batch_size * self.distance_num

                hat_amps = hat_target_amp[:D_batch_size]
                target_amps = hat_target_amp[D_batch_size:]
                hat_phases = hat_target_phs[:D_batch_size]
                target_phases = hat_target_phs[D_batch_size:]

                # discriminator_loss is not used in the validation
                validate_losses_tensor[-1] = 0.0

                loss_from_discriminator = -torch.mean(self.discriminator(hat_amps))

                _ = self.G_loss(
                    hat_amps,
                    target_amps,
                    hat_phases,
                    target_phases,
                    loss_from_discriminator,
                    validate_losses_tensor,
                )

                _ = self.record_metrics(
                    hat_amps,
                    target_amps,
                    validate_metrics_tensor,
                )

            validate_losses_tensor /= n_batch
            validate_metrics_tensor /= n_batch

        self.generator.train()
        self.discriminator.train()

        return validate_losses_tensor, validate_metrics_tensor

    def _add_losses_metrics_to_dict(
        self,
        epoch,
        n_batch_in_epoch,
        n_train,
        n_batch,
        validate_losses_tensor_iter,
        validate_metrics_tensor_iter,
        train_losses_tensor_iter,
        train_metrics_tensor_iter,
        recorder=None,
    ):
        recorder["epoch"].append(epoch)
        recorder["n_batch_in_epoch"].append(n_batch_in_epoch)
        recorder["n_train"].append(n_train)
        recorder["n_batch"].append(n_batch)

        recorder["train_losses_tensor"]["focal_phase_gradient_loss"].append(
            train_losses_tensor_iter[0].item()
        )
        recorder["train_losses_tensor"]["perceptual_loss"].append(
            train_losses_tensor_iter[1].item()
        )
        recorder["train_losses_tensor"]["pixel_loss"].append(
            train_losses_tensor_iter[2].item()
        )
        recorder["train_losses_tensor"]["TV_loss"].append(
            train_losses_tensor_iter[3].item()
        )
        recorder["train_losses_tensor"]["gan_loss"].append(
            train_losses_tensor_iter[4].item()
        )
        recorder["train_losses_tensor"]["G_loss"].append(
            train_losses_tensor_iter[5].item()
        )
        recorder["train_losses_tensor"]["D_loss"].append(
            train_losses_tensor_iter[6].item()
        )

        recorder["train_metrics_tensor"]["PSNR"].append(
            train_metrics_tensor_iter[0].item()
        )
        recorder["train_metrics_tensor"]["SSIM"].append(
            train_metrics_tensor_iter[1].item()
        )

        recorder["validate_losses_tensor"]["focal_phase_gradient_loss"].append(
            validate_losses_tensor_iter[0].item()
        )
        recorder["validate_losses_tensor"]["perceptual_loss"].append(
            validate_losses_tensor_iter[1].item()
        )
        recorder["validate_losses_tensor"]["pixel_loss"].append(
            validate_losses_tensor_iter[2].item()
        )
        recorder["validate_losses_tensor"]["TV_loss"].append(
            validate_losses_tensor_iter[3].item()
        )
        recorder["validate_losses_tensor"]["gan_loss"].append(
            validate_losses_tensor_iter[4].item()
        )
        recorder["validate_losses_tensor"]["G_loss"].append(
            validate_losses_tensor_iter[5].item()
        )
        recorder["validate_losses_tensor"]["D_loss"].append(
            validate_losses_tensor_iter[6].item()
        )

        recorder["validate_metrics_tensor"]["PSNR"].append(
            validate_metrics_tensor_iter[0].item()
        )
        recorder["validate_metrics_tensor"]["SSIM"].append(
            validate_metrics_tensor_iter[1].item()
        )

    def _save_losses_metrics_to_dict(self, loss_metrics_file):
        with open(loss_metrics_file, "w") as f:
            json.dump(self.dict_for_losses_metrics, f)

    def _input_dummy_tensor(self, input_shape):
        dummy_input = torch.clamp(
            torch.randn(*(1, 4, 192, 192), device="cuda"), min=0.1, max=0.9
        ).to(self.device)
        _ = self.generator(dummy_input)


class watermelon_without_GAN(watermelon):
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        distance_stack=torch.linspace(-1.5e-4, 0.0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        super(watermelon_without_GAN, self).__init__(
            filter_radius_coefficient=filter_radius_coefficient,
            pad_size=pad_size,
            distance_stack=distance_stack,
            pretrained_model_path_G=pretrained_model_path_G,
            pretrained_model_path_D=None,
            input_shape=input_shape,
            cuda=cuda,
        )

        self.discriminator = fakeDiscriminator(
            pretrained_model_path=None,
            feature_d=32,
            cuda=True,
        )

    def train(
        self,
        data_loader_train,
        data_loader_val,
        phs_gradient_loss_weight=1,
        perceptual_loss_weight=1.0,
        pixel_loss_weight=1.0,
        TV_loss_weight=1e-3,
        discriminator_loss_weight=1.0,
        epoch_num=2,
        lr_G=1e-3,
        lr_D=1e-3,
        save_path_G=None,
        save_path_D=None,
        info_print_interval=100,
        info_plot_interval=600,
        loss_metrics_file=None,
        save_path_img=None,
        checkpoint_iterval=5,
        discriminator_train_ratio=2,
        discriminator_lambda=10,
        step_scheduler_G_gamma=0.1,
        step_scheduler_D_gamma=0.9999,
        visualization_RGBD_AP=None,
    ):
        discriminator_loss_weight = 0.0
        discriminator_train_ratio = 0
        discriminator_lambda = 0.0
        super(watermelon_without_GAN, self).train(
            data_loader_train,
            data_loader_val,
            phs_gradient_loss_weight=phs_gradient_loss_weight,
            perceptual_loss_weight=perceptual_loss_weight,
            pixel_loss_weight=pixel_loss_weight,
            TV_loss_weight=TV_loss_weight,
            discriminator_loss_weight=discriminator_loss_weight,
            epoch_num=epoch_num,
            lr_G=lr_G,
            save_path_G=save_path_G,
            info_print_interval=info_print_interval,
            info_plot_interval=info_plot_interval,
            loss_metrics_file=loss_metrics_file,
            save_path_img=save_path_img,
            checkpoint_iterval=checkpoint_iterval,
            discriminator_train_ratio=discriminator_train_ratio,
            discriminator_lambda=discriminator_lambda,
            step_scheduler_G_gamma=step_scheduler_G_gamma,
            visualization_RGBD_AP=visualization_RGBD_AP,
        )


class watermelon_without_GAN_without_modulation(watermelon_without_GAN):
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        distance_stack=torch.linspace(-0.00015, 0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        super(watermelon_without_GAN_without_modulation, self).__init__(
            filter_radius_coefficient,
            pad_size,
            distance_stack,
            pretrained_model_path_G,
            pretrained_model_path_D,
            input_shape,
            cuda,
        )

        self.generator.part2.part1 = fakeChannelWiseSymmetricConv(
            kernel_size=3, padding=1
        ).to(self.device)


class watermelon_without_GAN_without_perceptual_loss(watermelon_without_GAN):
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        distance_stack=torch.linspace(-0.00015, 0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        super(watermelon_without_GAN_without_perceptual_loss, self).__init__(
            filter_radius_coefficient,
            pad_size,
            distance_stack,
            pretrained_model_path_G,
            pretrained_model_path_D,
            input_shape,
            cuda,
        )

    def G_loss(
        self,
        hat_amps,
        target_amps,
        hat_phs,
        target_phs,
        loss_from_discriminator,
        recorder=None,
    ):
        phs_loss = (
            focal_sincos_phase_gradient_loss(hat_phs, target_phs)
            * self.phs_gradient_loss_weight
        )
        pixel_loss = F.mse_loss(hat_amps, target_amps) * self.pixel_loss_weight
        TV_loss = total_variation_loss(hat_amps, target_amps) * self.TV_loss_weight
        gan_loss = loss_from_discriminator * self.discriminator_loss_weight
        loss = phs_loss + pixel_loss + TV_loss + gan_loss

        with torch.no_grad():
            recorder += torch.tensor(
                [phs_loss, 0.0, pixel_loss, TV_loss, gan_loss, loss, 0.0],
                device=self.device,
            )

        return loss


class watermelon_without_GAN_and_plain_phase_loss(watermelon_without_GAN):
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        distance_stack=torch.linspace(-0.00015, 0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        super(watermelon_without_GAN_and_plain_phase_loss, self).__init__(
            filter_radius_coefficient,
            pad_size,
            distance_stack,
            pretrained_model_path_G,
            pretrained_model_path_D,
            input_shape,
            cuda,
        )

    def G_loss(
        self,
        hat_amps,
        target_amps,
        hat_phs,
        target_phs,
        loss_from_discriminator,
        recorder=None,
    ):
        phs_loss = plain_phase_loss(hat_phs, target_phs) * self.phs_gradient_loss_weight
        perceptual_loss = (
            self.perceptual_loss(hat_amps, target_amps) * self.perceptual_loss_weight
        )
        pixel_loss = F.mse_loss(hat_amps, target_amps) * self.pixel_loss_weight
        TV_loss = total_variation_loss(hat_amps, target_amps) * self.TV_loss_weight
        gan_loss = loss_from_discriminator * self.discriminator_loss_weight
        loss = phs_loss + perceptual_loss + pixel_loss + TV_loss + gan_loss

        with torch.no_grad():
            recorder += torch.tensor(
                [phs_loss, perceptual_loss, pixel_loss, TV_loss, gan_loss, loss, 0.0],
                device=self.device,
            )

        return loss


class watermelon_without_GAN_and_focal_sincos_phase_loss(watermelon_without_GAN):
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        distance_stack=torch.linspace(-0.00015, 0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        super(watermelon_without_GAN_and_focal_sincos_phase_loss, self).__init__(
            filter_radius_coefficient,
            pad_size,
            distance_stack,
            pretrained_model_path_G,
            pretrained_model_path_D,
            input_shape,
            cuda,
        )

    def G_loss(
        self,
        hat_amps,
        target_amps,
        hat_phs,
        target_phs,
        loss_from_discriminator,
        recorder=None,
    ):
        phs_loss = (
            focal_sincos_phase_loss(hat_phs, target_phs) * self.phs_gradient_loss_weight
        )
        perceptual_loss = (
            self.perceptual_loss(hat_amps, target_amps) * self.perceptual_loss_weight
        )
        pixel_loss = F.mse_loss(hat_amps, target_amps) * self.pixel_loss_weight
        TV_loss = total_variation_loss(hat_amps, target_amps) * self.TV_loss_weight
        gan_loss = loss_from_discriminator * self.discriminator_loss_weight
        loss = phs_loss + perceptual_loss + pixel_loss + TV_loss + gan_loss

        with torch.no_grad():
            recorder += torch.tensor(
                [phs_loss, perceptual_loss, pixel_loss, TV_loss, gan_loss, loss, 0.0],
                device=self.device,
            )

        return loss


class watermelon_without_GAN_and_phase_sincos_gradient_loss(watermelon_without_GAN):
    def __init__(
        self,
        filter_radius_coefficient=0.5,
        pad_size=416,
        distance_stack=torch.linspace(-0.00015, 0, 8)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, 192, 192),
        cuda=True,
    ):
        super(watermelon_without_GAN_and_phase_sincos_gradient_loss, self).__init__(
            filter_radius_coefficient,
            pad_size,
            distance_stack,
            pretrained_model_path_G,
            pretrained_model_path_D,
            input_shape,
            cuda,
        )

    def G_loss(
        self,
        hat_amps,
        target_amps,
        hat_phs,
        target_phs,
        loss_from_discriminator,
        recorder=None,
    ):
        phs_loss = (
            phase_sincos_gradient_loss(hat_phs, target_phs)
            * self.phs_gradient_loss_weight
        )
        perceptual_loss = (
            self.perceptual_loss(hat_amps, target_amps) * self.perceptual_loss_weight
        )
        pixel_loss = F.mse_loss(hat_amps, target_amps) * self.pixel_loss_weight
        TV_loss = total_variation_loss(hat_amps, target_amps) * self.TV_loss_weight
        gan_loss = loss_from_discriminator * self.discriminator_loss_weight
        loss = phs_loss + perceptual_loss + pixel_loss + TV_loss + gan_loss

        with torch.no_grad():
            recorder += torch.tensor(
                [phs_loss, perceptual_loss, pixel_loss, TV_loss, gan_loss, loss, 0.0],
                device=self.device,
            )

        return loss
