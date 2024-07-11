import torch
import utilities


def generate_grid_x_y_wavelength(
    rowNum: int = 192,
    colNum: int = 192,
    wavelength: torch.Tensor = torch.tensor([639e-9, 515e-9, 473e-9]),
    pixel_pitch: float = 3.74e-6,
) -> torch.Tensor:
    x = (
        torch.linspace(0, (rowNum - 1) * pixel_pitch, rowNum)
        - (rowNum - 1) * pixel_pitch / 2
    )
    y = (
        torch.linspace(0, (colNum - 1) * pixel_pitch, colNum)
        - (colNum - 1) * pixel_pitch / 2
    )

    if rowNum % 2 == 0:
        x += pixel_pitch / 2
    if colNum % 2 == 0:
        y += pixel_pitch / 2

    grid_x_y = x.unsqueeze(1) ** 2 + y.unsqueeze(0) ** 2

    if wavelength.dim() == 0:
        grid_waveLength_x_y = torch.pi * grid_x_y / wavelength
    else:
        grid_waveLength_x_y = (
            torch.pi * grid_x_y.unsqueeze(0) / wavelength.unsqueeze(1).unsqueeze(2)
        )

    return grid_waveLength_x_y


def generate_grid_distance(
    grid_x_y_wavelength: torch.Tensor,
    distances: torch.Tensor = torch.Tensor([2.5e-3]),
) -> torch.Tensor:
    if distances.dim() == 0:
        grid_distance = torch.exp(1j * grid_x_y_wavelength / distances)
    else:
        grid_distance = torch.exp(
            1j
            * grid_x_y_wavelength.unsqueeze(0)
            / distances.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        )
    return grid_distance


def propagate(
    phaseTensor: torch.Tensor = None,
    grid: torch.Tensor = None,
    # distances: torch.Tensor = torch.Tensor([0.0]),
    # pixel_pitch=3.74e-6,
    # wave_length=torch.tensor([639e-9, 515e-9, 473e-9]),
) -> torch.Tensor:
    # intensity = utilities.intensity_calculator(
    #     torch.fft.fftshift(
    #         torch.fft.fft2(torch.exp(1j * phaseTensor) * grid), dim=(-1, -2)
    #     )
    # )
    intensity = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.exp(1j * phaseTensor) * grid),dim=(-1,-2))) ** 2

    return intensity
