import sys
import torch
from learnedMethodForHologram.bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod as BLASM_v2,
)
from learnedMethodForHologram.bandlimited_angular_spectrum_approach import (
    bandLimitedAngularSpectrumMethod_for_single_fixed_distance as BLASM_v3,
)


def test(distance=torch.tensor([12e-3])):
    propagator_v2 = BLASM_v2(
        sample_row_num=192,
        sample_col_num=192,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
    )

    w_grid_v2 = propagator_v2.generate_w_grid()
    transfer_function_v2 = propagator_v2.generate_transfer_function(distances=distance)
    # print(transfer_function_v2.shape)

    propagator_v3 = BLASM_v3(
        sample_row_num=192,
        sample_col_num=192,
        pixel_pitch=3.74e-6,
        wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
        band_limit=False,
        cuda=False,
        distance=distance,
    )

    w_grid_v3 = propagator_v3.generate_w_grid()
    transfer_function_v3 = propagator_v3.generate_transfer_function()
    # print(transfer_function_v3.shape)

    assert torch.allclose(w_grid_v2, w_grid_v3)
    assert torch.allclose(transfer_function_v2, transfer_function_v3)


if __name__ == "__main__":
    sys.exit(test())
