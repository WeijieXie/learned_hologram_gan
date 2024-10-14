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

    propagator = learnedMethodForHologram.angular_spectrum_method.bandLimitedAngularSpectrumMethod(
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
    )
    normalized_intensities = learnedMethodForHologram.utilities.tensor_normalizor_2D(
        intensities
    )

    # learnedMethodForHologram.utilities.multi_sample_plotter(
    #     normalized_intensities,
    #     titles=None,
    #     rgb_img=True,
    #     save_dir=output_directory,
    #     color=0,
    # )


if __name__ == "__main__":
    sys.exit(test())
