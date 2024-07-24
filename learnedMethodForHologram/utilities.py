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


def cut_center_256_192(tensor_256):
    """
    Cut the last 2 dims of the tensor from doubled size to the original size

    Args:
    tensorX: x-D tensor, the tensor to be cut

    Returns:
    tensorX: x-D tensor, the cut tensor
    """
    tensor_192 = tensor_256[
        ...,
        32:223,
        32:223,
    ]
    return tensor_192


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
    tensor_to_plot,
    title=None,
    save_dir=None,
    rgb_img=True,
    color=0,  # r = 0, g = 1, b = 2
):
    """
    Plot the tensor as an RGB image

    Args:
    tensor_to_plot: 2-D or 3-D tensor or 4-D tensor(0 or 1 in the first dimension), the tensor to plot
    title: string, the title of the plot
    save_dir: string, the directory to save the plot
    rgb_img: bool, whether to plot the RGB image
    color: int, the color channel to plot, only used when the tensor can be squeeze to 2-D

    Returns:
    None
    """
    tensor_to_plot = tensor_to_plot.squeeze().to("cpu")

    if tensor_to_plot.dim() >= 4 or tensor_to_plot.dim() <= 1:
        raise ValueError(
            "Only 2-D and 3-D tensors are supported. The input tensor is {}-D.".format(
                tensor_to_plot.dim()
            )
        )

    if title is None:
        title = "title_not_provided"

    elif tensor_to_plot.dim() == 2:
        rgb_tensor = torch.empty(3, tensor_to_plot.shape[-2], tensor_to_plot.shape[-1])
        rgb_tensor[color] = tensor_to_plot
        rgb_tensor = rgb_tensor.permute(1, 2, 0)

        plt.figure()
        plt.imshow(rgb_tensor)
        plt.axis("off")
        plt.title(title)
        plt.show()

        if save_dir.isinstance(str):
            color = ["red", "green", "blue"][color]
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"{title}_{color}.png",
                )
            )

    # if the input tensor is a 3-D tensor
    else:
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rgb_img = tensor_to_plot.permute(1, 2, 0).numpy()
            plt.imsave(
                os.path.join(save_dir, f"{title}.png"),
                rgb_img,
            )
        else:
            if tensor_to_plot.shape[0] != 3:
                raise ValueError(
                    "The input tensor should have 3 channels to represent RGB. The input tensor has {} channels.".format(
                        tensor_to_plot.shape[0]
                    )
                )

            if rgb_img:
                fig, axs = plt.subplots(1, 4, figsize=(30, 15))
                axs[3].imshow(tensor_to_plot.permute(1, 2, 0))
                axs[3].axis("off")
                axs[3].set_title(title)
            else:
                fig, axs = plt.subplots(1, 3, figsize=(30, 15))

            for i in range(3):
                rgb_tensor = torch.zeros(
                    3, tensor_to_plot.shape[-2], tensor_to_plot.shape[-1]
                )
                rgb_tensor[i] = tensor_to_plot[i]
                rgb_tensor = rgb_tensor.permute(1, 2, 0)
                axs[i].imshow(rgb_tensor)
                axs[i].axis("off")
                # keep 3 decimal places
                axs[i].set_title(title)
        plt.show()


def multi_sample_plotter(
    tensor_to_plot,
    titles=None,
    rgb_img=True,
    save_dir=None,
    color=0,  # r = 0, g = 1, b = 2
):
    if titles is None:
        titles = range(len(tensor_to_plot))
    tensor_to_plot.to("cpu")
    for i in range(tensor_to_plot.shape[-4]):
        multi_channel_plotter(tensor_to_plot[i], titles[i], save_dir, rgb_img, color)


def generate_circular_frequency_mask(
    sample_row_num=192,
    sample_col_num=192,
    radius=60,
    decay_rate=None,
):
    """
    Generate the circular frequency mask where the radius means the radius on the shorter edge,
    the function will keep the radius on the longer edge to the longer edge the same portion as the radius on the shorter edge to the shorter edge

    Args:
    sample_row_num: int, the number of rows of the mask
    sample_col_num: int, the number of columns of the mask
    radius: float, the radius of the circular mask
    decay_rate: float, the decay rate outside the circular mask

    Returns:
    mask: 2-D tensor, the circular frequency mask
    """

    shorter_edge = min(sample_row_num, sample_col_num)
    if radius > shorter_edge / 2:
        raise ValueError(
            f"The radius {radius} is larger than the half of the sample size {shorter_edge/2}"
        )

    # Create a grid of (u, v) coordinates
    u = torch.fft.fftfreq(sample_row_num).unsqueeze(-1)
    v = torch.fft.fftfreq(sample_col_num).unsqueeze(0)
    D = torch.sqrt(u**2 + v**2) * shorter_edge

    mask = torch.ones_like(D)
    if decay_rate is not None:
        # Create the circular low-pass filter with exponential decay around the radius
        mask[D > radius] = torch.exp(-decay_rate * (D[D > radius] - radius))
    else:
        mask[D > radius] = 0.0

    return mask


def generate_square_frequency_mask(
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


def gpu_timer(operation, repeat=100):
    total_time = 0
    # clean the cache on gpu
    torch.cuda.empty_cache()
    for _ in range(repeat):
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        operation()
        end_time.record()

        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
    return total_time / repeat


def unzip_file(zip_path, dest_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)
