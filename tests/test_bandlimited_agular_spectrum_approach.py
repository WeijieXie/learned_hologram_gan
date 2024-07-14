import sys
import torch
import learnedMethodForHologram.utilities


def test(output_directory="output/test_output"):
    phase_tensor = learnedMethodForHologram.utilities.phase_tensor_generator(
        "data/images/sample_hologram.png"
    )
    amplitude_tensor = torch.ones_like(phase_tensor)
    distances = torch.linspace(0, 5e-3, 7)
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
        padding=False,
        cuda=False,
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
    learnedMethodForHologram.utilities.multi_depth_plotter(
        normalized_intensities,
        distances,
        rgb_img=True,
        save_dir=output_directory,
        color=0,
    )


if __name__ == "__main__":
    sys.exit(test())
