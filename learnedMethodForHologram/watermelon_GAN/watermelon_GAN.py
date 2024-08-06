import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.autograd as autograd


from .generator import Generator
from .discriminator import WGANGPDiscriminator192
from .loss_func import PerceptualLoss, total_variation_loss, focal_phase_gradient_loss

from ..bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_multiple_distances,
)

from ..utilities import try_gpu, multi_sample_plotter, tensor_normalizor_2D


class watermelon_gan:
    def __init__(
        self,
        filter_radius_coefficient=0.5,
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
            sample_row_num=192,
            sample_col_num=192,
            pad_size=160,
            pretrained_model_path=pretrained_model_path_G,
            pixel_pitch=3.74e-6,
            wave_length=torch.tensor([638e-9, 520e-9, 450e-9]),
            distance=torch.tensor([1e-3]),
        )

        self.discriminator = WGANGPDiscriminator192(
            pretrained_model_path=pretrained_model_path_D,
            cuda=True,
        )

        self.perceptual_loss = PerceptualLoss(
            feature_map_layers=[3, 8, 13, 22, 31],
            cuda=True,
        )

        self.propagator = bandLimitedAngularSpectrumMethod_for_multiple_distances(
            sample_row_num=192,
            sample_col_num=192,
            distances=distance_stack,
            pad_size=160,
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

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D)

        step_scheduler_G = ReduceLROnPlateau(
            optimizer_G,
            "min",
            factor=step_scheduler_G_gamma,
            patience=5,
            verbose=True,
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=1e-6,
        )

        step_scheduler_D = ReduceLROnPlateau(
            optimizer_D,
            "min",
            factor=step_scheduler_D_gamma,
            patience=100,
            verbose=True,
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=1e-6,
        )

        for epoch in range(epoch_num):

            with torch.no_grad():
                self.train_losses_tensor = torch.zeros(7).to(self.device)
                self.validate_losses_tensor = torch.zeros(7).to(self.device)
                n_train = 0
                n_val = 0

                train_losses_tensor_last = torch.zeros(7).to(self.device)
                n_train_last = 0

            self.generator.train()
            self.discriminator.train()

            for iter_num, (RGBD, target_amp, target_phs) in enumerate(
                data_loader_train
            ):

                n_train += RGBD.size(0)
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
                    self.propagator.propagate_fixed_multiple_distances_freq2amp_SD(
                        hat_target_freq
                    )
                )

                # the hat and target amplitude at multible distances
                D_batch_size = G_batch_size * self.distance_num

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
                    if iter_num % info_print_interval == 0:
                        train_losses_tensor_iter = (
                            self.train_losses_tensor - train_losses_tensor_last
                        ) / (n_train - n_train_last)

                        print(
                            f"epoch {epoch}, batch {iter_num + 1}:\n"
                            f"train: focal_phase_gradient_loss {train_losses_tensor_iter[0]}, perceptual_loss {train_losses_tensor_iter[1]}, pixel_loss {train_losses_tensor_iter[2]}, TV_loss {train_losses_tensor_iter[3]}, gan_loss {train_losses_tensor_iter[4]}, G_loss {train_losses_tensor_iter[5]}, D_loss {train_losses_tensor_iter[6]}"
                        )
                        train_losses_tensor_last = self.train_losses_tensor.clone()
                        n_train_last = n_train

                    if iter_num % info_plot_interval == 0:
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
                                    f"amp_hat in epoch {epoch}, batch {iter_num + 1}",
                                    f"phs_hat in epoch {epoch}, batch {iter_num + 1}",
                                ],
                                rgb_img=True,
                                save_dir=save_path_img,
                            )
                            print(
                                f"visualization saved at epoch {epoch}, batch {iter_num + 1}"
                            )

            self.generator.eval()
            self.discriminator.eval()

            for i, (RGBD, target_amp, target_phs) in enumerate(data_loader_val):

                with torch.no_grad():

                    n_val += RGBD.size(0)
                    G_batch_size = RGBD.size(0)

                    # phase at z = z0
                    POH_phs = self.generator(RGBD)

                    # filtered frequency of hat at z = z1
                    hat_freq = (
                        self.generator.part2.propagator.propagate_POH2Freq_forward(
                            POH_phs
                        )
                    )

                    # filtered frequency of target at z = z1
                    target_freq = self.propagator.filter_AP2filteredFreq(
                        target_amp, target_phs
                    )

                    # propagate the hat and target frequency to different distances
                    hat_target_freq = torch.cat((hat_freq, target_freq), dim=0)
                    hat_target_amp, hat_target_phs = (
                        self.propagator.propagate_fixed_multiple_distances_freq2amp_SD(
                            hat_target_freq
                        )
                    )

                    # the hat and target amplitude at multible distances
                    D_batch_size = G_batch_size * self.distance_num

                    hat_amps = hat_target_amp[:D_batch_size]
                    target_amps = hat_target_amp[D_batch_size:]
                    hat_phases = hat_target_phs[:D_batch_size]
                    target_phases = hat_target_phs[D_batch_size:]

                    real_validity = self.discriminator(target_amps)
                    fake_validity = self.discriminator(hat_amps)

                with torch.enable_grad():
                    gradient_penalty = self.compute_gradient_penalty(
                        target_amps, hat_amps
                    )

                with torch.no_grad():
                    discriminator_loss = (
                        -torch.mean(real_validity) + torch.mean(fake_validity)
                    ) + discriminator_lambda * gradient_penalty

                    self.validate_losses_tensor[-1] += discriminator_loss

                    loss_from_discriminator = -torch.mean(self.discriminator(hat_amps))

                    generator_loss = self.G_loss(
                        hat_amps,
                        target_amps,
                        hat_phases,
                        target_phases,
                        loss_from_discriminator,
                        self.validate_losses_tensor,
                    )

            self.train_losses_tensor /= n_train
            self.validate_losses_tensor /= n_val

            print(
                f"epoch {epoch + 1}:\n"
                f"train: focal_phase_gradient_loss {self.train_losses_tensor[0]}, perceptual_loss {self.train_losses_tensor[1]}, pixel_loss {self.train_losses_tensor[2]}, TV_loss {self.train_losses_tensor[3]}, gan_loss {self.train_losses_tensor[4]}, G_loss {self.train_losses_tensor[5]}, D_loss {self.train_losses_tensor[6]}\n"
                f"validate: focal_phase_gradient_loss {self.validate_losses_tensor[0]}, perceptual_loss {self.validate_losses_tensor[1]}, pixel_loss {self.validate_losses_tensor[2]}, TV_loss {self.validate_losses_tensor[3]}, gan_loss {self.validate_losses_tensor[4]}, G_loss {self.validate_losses_tensor[5]}, D_loss {self.validate_losses_tensor[6]}"
            )

            # step_scheduler_G.step(self.validate_losses_tensor[4])
            # step_scheduler_D.step(self.validate_losses_tensor[5])

            # checkpoint
            if epoch % checkpoint_iterval == 0 and epoch != 0:
                if save_path_G is not None:
                    check_point_path = save_path_G.replace(".pth", f"_epoch{epoch}.pth")
                    torch.save(self.generator.state_dict(), check_point_path)
                    print(f"Generator saved to {check_point_path}")

                if save_path_D is not None:
                    check_point_path = save_path_D.replace(".pth", f"_epoch{epoch}.pth")
                    torch.save(self.discriminator.state_dict(), check_point_path)
                    print(f"Discriminator saved to {check_point_path}")

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
            focal_phase_gradient_loss(hat_phs, target_phs)
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

    def _input_dummy_tensor(self, input_shape):
        dummy_input = torch.clamp(
            torch.randn(*(1, 4, 192, 192), device="cuda"), min=0.1, max=0.9
        ).to(self.device)
        _ = self.generator(dummy_input)
