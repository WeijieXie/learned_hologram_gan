import sys
import torch
import learnedMethodForHologram.utilities


def test(output_directory="output/test_output", cuda=False):
    device = (
        learnedMethodForHologram.utilities.try_gpu() if cuda else torch.device("cpu")
    )
    phase_tensor = learnedMethodForHologram.utilities.phase_tensor_generator(
        "data/images/sample_hologram.png"
    ).to(device)
    amplitude_tensor = torch.ones_like(phase_tensor).to(device)
    distances = torch.linspace(-1e-3, 2.5e-3, 4).to(device)
    spacial_frequency_filter = (
        learnedMethodForHologram.utilities.generate_custom_frequency_mask(
            sample_row_num=2400,
            sample_col_num=4094,
            x=800,
            y=int(800 * 4094 / 2400),
        )
    )
    propagator = learnedMethodForHologram.bandlimited_agular_spectrum_approach.bandLimitedAngularSpectrumMethod(
        sample_row_num=2400,
        sample_col_num=4094,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=True,
        cuda=cuda,
    )
    intensities = propagator(
        amplitute_tensor=amplitude_tensor,
        phase_tensor=phase_tensor,
        distances=distances,
        spacial_frequency_filter=spacial_frequency_filter,
    )
    normalized_intensities = learnedMethodForHologram.utilities.tensor_normalizor_2D(
        intensities
    )

    # print(normalized_intensities.shape)
    # max, _ = torch.max(normalized_intensities, dim=-1, keepdim=True)
    # max, _ = torch.max(max, dim=-2, keepdim=True)
    # min, _ = torch.min(normalized_intensities, dim=-1, keepdim=True)
    # min, _ = torch.min(min, dim=-2, keepdim=True)
    # print(max.shape, min.shape)
    # print(max.squeeze())
    # print(min.squeeze())

    learnedMethodForHologram.utilities.multi_depth_plotter(
        normalized_intensities,
        distances,
        rgb_img=True,
        save_dir=output_directory,
        color=0,
    )


if __name__ == "__main__":
    sys.exit(test())
