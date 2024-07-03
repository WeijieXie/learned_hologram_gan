import math
import numpy as np
import torch
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt


def complex_plain(amplitude_tensor, phase_tensor):
    """
    Generate the complex tensor from the amplitude and phase tensor

    Args:
    amplitude_tensor: 2-D or 3-D tensor, the amplitude tensor
    phase_tensor: 2-D or 3-D tensor, the phase tensor

    Returns:
    source: 2-D or 3-D tensor, the complex tensor
    """
    complex_tensor = amplitude_tensor * torch.exp(1j * phase_tensor)
    return complex_tensor


def amplitude_tensor_generator_for_phase_only_hologram(image_path):
    """
    Generate the uniform amplitude tensor the same size as the phase tensor  with all values are 1.0

    Args:
    image_path: string, the path of the image

    Returns:
    amplitude_tensor: 2-D tensor, the amplitude tensor
    """
    image = Image.open(image_path)
    width = image.size[0]
    height = image.size[1]
    # amplitude_tensor = torch.ones([3, height, width])
    amplitude_tensor = torch.ones([*image.size])
    return amplitude_tensor


def phase_tensor_generator(image_path):
    """
    Generate the phase tensor from the image with values normalized to 0-2*pi

    Args:
    image_path: string, the path of the image

    Returns:
    phase_tensor: 2-D tensor, the phase tensor
    """
    image = Image.open(image_path)  # convert the image to gray scale
    transform = transforms.ToTensor()
    image_tensor = transform(image)  # convert the image to tensor
    image_tensor_normalized = image_tensor * 2 * math.pi
    return image_tensor_normalized


def zero_padding(tensorX):
    """
    Pad the last 2 dims of the tensor to double the height and width

    Args:
    tensorX: x-D tensor, the tensor to be padded

    Returns:
    tensorX: x-D tensor, the padded tensor
    """
    tensorX = torch.nn.functional.pad(
        tensorX,
        (
            tensorX.shape[-1] // 2,
            tensorX.shape[-1] // 2,
            tensorX.shape[-2] // 2,
            tensorX.shape[-2] // 2,
        ),
        mode="constant",
        value=0,
    )
    return tensorX


def cut_padding(tensorX):
    """
    Cut the last 2 dims of the tensor from doubled size to the original size

    Args:
    tensorX: x-D tensor, the tensor to be cut

    Returns:
    tensorX: x-D tensor, the cut tensor
    """
    tensorX = tensorX[
        ...,
        tensorX.shape[-1] // 4 : (3 * tensorX.shape[-1]) // 4,
        tensorX.shape[-2] // 4 : (3 * tensorX.shape[-2]) // 4,
    ]
    return tensorX


# def intensity_normalizor(intensity):
#     """
#     Normalize the 2-D intensity tensor to 0-255

#     Args:
#     intensity: 2-D tensor, the intensity tensor

#     Returns:
#     intensity_normalized: 2-D tensor, the normalized intensity tensor
#     """
#     intensity_normalized = (
#         255 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
#     )
#     return intensity_normalized


def amplitude_calculator(complex_tensor):
    """
    Calculate the amplitude from the complex tensor

    Args:
    complex_tensor: x-D tensor, the complex tensor

    Returns:
    amplitude: x-D tensor, the amplitude tensor
    """
    amplitude = torch.abs(complex_tensor)
    return amplitude


def phase_calculator(complex_tensor):
    """
    Calculate the phase from the complex tensor

    Args:
    complex_tensor: x-D tensor, the complex tensor

    Returns:
    phase: x-D tensor, the phase tensor
    """
    phase = torch.angle(complex_tensor)
    return phase


def intensity_calculator(complex_tensor, intensity_norm=False):
    """
    Calculate the intensity from the complex tensor
    """
    intensity = complex_tensor.abs() ** 2
    # if intensity_norm:
    #     intensity = intensity_normalizor(intensity)
    return intensity


def diffraction_plotter(
    diffraction_tensor,
    distance,
    rgb_img=False,
    color=0,  # r = 0, g = 1, b = 2
):
    """
    Plot the diffraction pattern

    Args:
    diffraction_tensor: 2-D or 3-D tensor, the diffraction pattern tensor
    distance: float, the distance of the diffraction pattern
    rgb_img: bool, whether to plot the RGB image
    color: int, the color channel to plot

    Returns:
    None
    """
    diffraction_tensor = diffraction_tensor.squeeze()

    if diffraction_tensor.dim() >= 4 or diffraction_tensor.dim() <= 1:
        raise ValueError(
            "Only 2-D and 3-D tensors are supported. The input tensor is {}-D.".format(
                diffraction_tensor.dim()
            )
        )

    elif diffraction_tensor.dim() == 2:
        rgb_tensor = torch.empty(
            3, diffraction_tensor.shape[-2], diffraction_tensor.shape[-1]
        )
        rgb_tensor[color] = diffraction_tensor
        rgb_tensor = rgb_tensor.permute(1, 2, 0)

        plt.figure()
        plt.imshow(rgb_tensor)
        plt.axis("off")
        plt.title("The diffraction pattern at z = {} mm".format(distance))
        plt.show()

    # if the input tensor is a 3-D tensor
    else:
        if diffraction_tensor.shape[0] != 3:
            raise ValueError(
                "The input tensor should have 3 channels to represent RGB. The input tensor has {} channels.".format(
                    diffraction_tensor.shape[0]
                )
            )

        if rgb_img:
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            axs[3].imshow(diffraction_tensor.permute(1, 2, 0))
            axs[3].axis("off")
            axs[3].set_title("The diffraction pattern at z = {} mm".format(distance))
        else:
            fig, axs = plt.subplots(1, 3, figsize=(20, 10))

        for i in range(3):
            rgb_tensor = torch.zeros(
                3, diffraction_tensor.shape[-2], diffraction_tensor.shape[-1]
            )
            rgb_tensor[i] = diffraction_tensor[i]
            rgb_tensor = rgb_tensor.permute(1, 2, 0)
            axs[i].imshow(rgb_tensor)
            axs[i].axis("off")
            axs[i].set_title("The diffraction pattern at z = {} mm".format(distance))
        plt.show()


def num_gpus():
    """Get the number of GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def try_gpu(i=0):
    if num_gpus() > i:
        return torch.device(f"cuda:{i}")
    else:
        print(f"gpu with index '{i}' is not available")
        return torch.device("cpu")


def try_all_gpus():
    """
    Return all available GPUs, or [cpu(),] if no GPU exists.

    Returns:
    -------
    devices: list
        A list of devices.
    """

    return [torch.device(f"cuda:{i}") for i in range(num_gpus())]


def gpus_info(gpu_list):
    for i, gpu in enumerate(gpu_list):
        print(f"""gpu {i}: {torch.cuda.get_device_name(gpu)}""")


def current_gpu_info():
    current_device = torch.cuda.current_device()
    print(f"""current gpu : {torch.cuda.get_device_name(current_device)}""")