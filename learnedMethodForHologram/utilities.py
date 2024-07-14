import math
import numpy as np
import torch
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import zipfile
import os


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


def amplitude_tensor_generator_for_phase_only_hologram(image_path_or_tensor):
    """
    Generate the uniform amplitude tensor the same size as the phase tensor  with all values are 1.0

    Args:
    image_path: string, the path of the image

    Returns:
    amplitude_tensor: 2-D tensor, the amplitude tensor
    """

    if isinstance(image_path_or_tensor, str):
        image = Image.open(image_path_or_tensor)
        width = image.size[0]
        height = image.size[1]
        amplitude_tensor = torch.ones([3, height, width])
        return amplitude_tensor
    elif isinstance(image_path_or_tensor, torch.Tensor):
        amplitude_tensor = torch.ones_like(image_path_or_tensor)
        return amplitude_tensor
    else:
        raise ValueError("The input should be a string or a tensor.")


def phase_tensor_generator(image_path_or_tensor):
    """
    Generate the phase tensor from the image with values normalized to 0-2*pi

    Args:
    image_path: string, the path of the image

    Returns:
    phase_tensor: 2-D tensor, the phase tensor
    """
    if isinstance(image_path_or_tensor, str):
        image = Image.open(image_path_or_tensor)  # convert the image to gray scale
        transform = transforms.ToTensor()
        image_tensor = transform(image)  # convert the image to tensor
        image_tensor_normalized = image_tensor * 2 * math.pi
        return image_tensor_normalized
    elif isinstance(image_path_or_tensor, torch.Tensor):
        # image_tensor_normalized = image_path_or_tensor * 2 * math.pi
        return image_path_or_tensor
    else:
        raise ValueError("The input should be a string or a tensor.")


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


def tensor_normalizor_2D(intensity):
    """
    Normalize the 2-D intensity tensor to 0-255

    Args:
    intensity: 2-D tensor, the intensity tensor

    Returns:
    intensity_normalized: 2-D tensor, the normalized intensity tensor
    """
    max, _ = torch.max(intensity, dim=-1, keepdim=True)
    max, _ = torch.max(max, dim=-2, keepdim=True)
    min, _ = torch.min(intensity, dim=-1, keepdim=True)
    min, _ = torch.min(min, dim=-2, keepdim=True)
    tensor_normalized = (intensity - min) / (max - min)
    return tensor_normalized


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


def intensity_calculator(complex_tensor, intensity_norm=True):
    """
    Calculate the intensity from the complex tensor
    """
    intensity = complex_tensor.abs() ** 2
    if intensity_norm:
        intensity = tensor_normalizor_2D(intensity)
    return intensity


def multi_channel_plotter(
    diffraction_tensor,
    distance,
    rgb_img=False,
    save_dir=None,
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
    diffraction_tensor = diffraction_tensor.squeeze().to("cpu")

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

        if save_dir.isinstance(str):
            color = ["red", "green", "blue"][color]
            plt.savefig(
                os.path.join(
                    save_dir, f"diffraction_at_{round(distance.item(), 5)}_{color}.png"
                )
            )

    # if the input tensor is a 3-D tensor
    else:
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rgb_img = diffraction_tensor.permute(1, 2, 0).numpy()
            plt.imsave(
                os.path.join(save_dir, "diffraction_at_{}.png".format(distance)),
                rgb_img,
            )
        else:
            if diffraction_tensor.shape[0] != 3:
                raise ValueError(
                    "The input tensor should have 3 channels to represent RGB. The input tensor has {} channels.".format(
                        diffraction_tensor.shape[0]
                    )
                )

            if rgb_img:
                fig, axs = plt.subplots(1, 4, figsize=(30, 15))
                axs[3].imshow(diffraction_tensor.permute(1, 2, 0))
                axs[3].axis("off")
                axs[3].set_title(
                    "The diffraction pattern at z = {} mm".format(round(distance.item(), 3))
                )
            else:
                fig, axs = plt.subplots(1, 3, figsize=(30, 15))

            for i in range(3):
                rgb_tensor = torch.zeros(
                    3, diffraction_tensor.shape[-2], diffraction_tensor.shape[-1]
                )
                rgb_tensor[i] = diffraction_tensor[i]
                rgb_tensor = rgb_tensor.permute(1, 2, 0)
                axs[i].imshow(rgb_tensor)
                axs[i].axis("off")
                # keep 3 decimal places
                axs[i].set_title(
                    "The diffraction pattern at z = {}. mm".format(
                        round(distance.item(), 3)
                    )
                )
        plt.show()


def multi_depth_plotter(
    diffraction_tensor,
    distances,
    rgb_img=False,
    save_dir=None,
    color=0,  # r = 0, g = 1, b = 2
):
    if save_dir is not None:
        distances = range(len(distances))
    diffraction_tensor.to("cpu")
    for i in range(diffraction_tensor.shape[-4]):
        multi_channel_plotter(
            diffraction_tensor[i], distances[i], rgb_img, save_dir, color
        )


def generate_custom_frequency_mask(
    sample_row_num=192,
    sample_col_num=192,
    x=0,
    y=0,
):
    if 2 * x > sample_row_num or 2 * y > sample_col_num:
        raise ValueError("The mask size is too large.")
    mask = torch.zeros((sample_row_num, sample_col_num))
    mask[
        sample_row_num // 2 - x : sample_row_num // 2 + 1 + x,
        sample_col_num // 2 - y : sample_col_num // 2 + 1 + y,
    ] = 1
    return torch.fft.ifftshift(mask)


def mask_generator(
    sample_row_num,
    sample_col_num,
    u_limit,
    v_limit,
    pixel_pitch=3.74e-6,
):
    """
    Generate the band limited mask

    Args:
    sample_row_num: int, the number of rows of the mask
    sample_col_num: int, the number of columns of the mask
    u_limit: float, the maximum frequency in the x direction
    v_limit: float, the maximum frequency in the y direction

    Returns:
    mask: 2-D tensor, the band limited mask
    """
    freq_x = torch.fft.fftfreq(sample_row_num, 1.0 / sample_row_num)
    freq_y = torch.fft.fftfreq(sample_col_num, 1.0 / sample_row_num)
    mask_u = torch.abs(freq_x) < torch.tensor(u_limit)
    mask_v = torch.abs(freq_y) < torch.tensor(v_limit)
    mask = mask_u.unsqueeze(1) & mask_v.unsqueeze(0)
    return mask


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


def unzip_file(zip_path, dest_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)
