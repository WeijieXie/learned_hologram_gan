import math
import numpy as np
import torch
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import random

import zipfile
import os
import json


def complex_plain(amplitude_tensor, phase_tensor):
    """
    Generate the complex tensor from the amplitude and phase tensor

    Args:
    amplitude_tensor: tensor, the amplitude tensor
    phase_tensor: tensor, the phase tensor

    Returns:
    complex_tensor: tensor, the complex tensor
    """
    complex_tensor = amplitude_tensor * torch.exp(1j * phase_tensor)
    return complex_tensor


def phase_tensor_generator(image_path_or_tensor):
    """
    Generate the phase tensor from the image path or tensor

    Args:
    image_path_or_tensor: string or tensor, the image path or tensor

    Returns:
    phase_tensor: tensor, the phase tensor
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


def amplitude_normalizor(amp):
    """
    Normalize the amplitude tensor to 0-1

    Args:
    amp: tensor, the amplitude tensor

    Returns:
    amp: tensor, the normalized amplitude tensor
    """
    max, _ = torch.max(amp, dim=-1, keepdim=True)
    max, _ = torch.max(max, dim=-2, keepdim=True)
    amp = amp / (max * 1.01)
    return amp


def tensor_normalizor_2D(tensor_to_normalize):
    """
    Normalize the tensor to 0-1 by the maximum and minimum value in each channel

    Args:
    intensity: tensor, the intensity tensor

    Returns:
    tensor_normalized: tensor, the normalized tensor
    """
    max, _ = torch.max(tensor_to_normalize, dim=-1, keepdim=True)
    max, _ = torch.max(max, dim=-2, keepdim=True)
    min, _ = torch.min(tensor_to_normalize, dim=-1, keepdim=True)
    min, _ = torch.min(min, dim=-2, keepdim=True)
    tensor_normalized = (tensor_to_normalize - min) / (max - min)
    return tensor_normalized


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

    elif tensor_to_plot.dim() == 2:

        if title is None:
            title = "title_not_provided"

        rgb_tensor = torch.zeros(3, tensor_to_plot.shape[-2], tensor_to_plot.shape[-1])
        rgb_tensor[color] = tensor_to_plot
        rgb_tensor = rgb_tensor.permute(1, 2, 0)

        plt.figure()
        # plt.imshow(rgb_tensor)
        plt.imshow(tensor_to_plot, cmap="gray")
        plt.axis("off")
        plt.title(title)
        plt.show()

        if save_dir is not None:
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
                axs[i].set_title(title)
        plt.show()


def multi_sample_plotter(
    tensor_to_plot,
    titles=None,
    rgb_img=True,
    save_dir=None,
    color=0,  # r = 0, g = 1, b = 2
):
    """
    Plot the multi-channel tensor

    Args:
    tensor_to_plot: 4-D tensor, the tensor to plot
    titles: list of strings, the titles of the plot
    rgb_img: bool, whether to plot the RGB image
    save_dir: string, the directory to save the plot
    color: int, the color channel to plot, only used when the tensor can be squeeze to 2-D

    Returns:
    None
    """
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


def generate_circular_frequency_mask_modified(
    sample_row_num=192,
    sample_col_num=192,
    filter_radius_coefficient=0.5,
):
    """
    Generate the circular frequency mask faster without the decay rate, the exception and the if statement

    Args:
    sample_row_num: int, the number of rows of the mask
    sample_col_num: int, the number of columns of the mask
    filter_radius_coefficient: float, the coefficient of the radius of the circular mask

    Returns:
    mask: 2-D tensor, the circular frequency mask
    """
    shorter_edge = min(sample_row_num, sample_col_num)
    radius = shorter_edge * filter_radius_coefficient

    # Create a grid of (u, v) coordinates
    u = torch.fft.fftfreq(sample_row_num).unsqueeze(-1)
    v = torch.fft.fftfreq(sample_col_num).unsqueeze(0)
    D = torch.sqrt(u**2 + v**2) * shorter_edge

    mask = torch.ones_like(D)
    mask[D > radius] = 0.0

    return mask


def prepare_circular_frequency_mask_grid(
    samplingRowNum,
    samplingColNum,
):
    """
    Prepare the circular frequency mask grid

    Args:
    samplingRowNum: int, the number of rows of the mask
    samplingColNum: int, the number of columns of the mask

    Returns:
    D: tensor, the circular frequency mask grid
    """
    shorter_edge = min(samplingRowNum, samplingColNum)

    # Create a grid of (u, v) coordinates
    u = torch.fft.fftfreq(samplingRowNum).unsqueeze(-1)
    v = torch.fft.fftfreq(samplingColNum).unsqueeze(0)
    D = torch.sqrt(u**2 + v**2) * shorter_edge
    return D


def generate_square_frequency_mask(
    sample_row_num=192,
    sample_col_num=192,
    x=0,
    y=0,
):
    """
    Generate the square frequency mask

    Args:
    sample_row_num: int, the number of rows of the mask
    sample_col_num: int, the number of columns of the mask
    x: int, the half length of the mask in the x direction
    y: int, the half length of the mask in the y direction

    Returns:
    mask: 2-D tensor, the square frequency mask
    """
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


def generate_checkerboard_mask(
    height=192,
    width=192,
    cell_size=4,
    reserve=False,
):
    """
    Generate the checkerboard mask

    Args:
    height: int, the height of the mask
    width: int, the width of the mask
    cell_size: int, the size of the cell
    reserve: bool, whether to reserve the 0 and 1

    Returns:
    checkerboard_torch: tensor, the checkerboard mask
    """
    x = np.arange(width).reshape(1, -1) // cell_size
    y = np.arange(height).reshape(-1, 1) // cell_size
    checkerboard_np = (x + y) % 2
    checkerboard_np = checkerboard_np.astype(np.float32)

    checkerboard_torch = torch.tensor(checkerboard_np)

    if reserve:
        checkerboard_torch = 1 - checkerboard_torch

    return checkerboard_torch


def set_seed(seed):
    """
    Set the seed for reproducibility

    Args:
    seed: int, the seed

    Returns:
    None"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    """
    Print the information of the GPUs
    """
    for i, gpu in enumerate(gpu_list):
        print(f"""gpu {i}: {torch.cuda.get_device_name(gpu)}""")


def current_gpu_info():
    """
    Print the information of the current GPU
    """
    current_device = torch.cuda.current_device()
    print(f"""current gpu : {torch.cuda.get_device_name(current_device)}""")


def gpu_timer(operation, repeat=100):
    """
    Measure the time of the operation on GPU

    Args:
    operation: function, the operation to measure
    repeat: int, the number of repeats

    Returns:
    total_time: float, the total time
    """
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
    """
    Unzip the file

    Args:
    zip_path: string, the path of the zip file
    dest_path: string, the path to extract the file

    Returns:
    None
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def extract_nested_value(data, keys):
    """
    Extract the nested value from the dictionary

    Args:
    data: dict, the dictionary
    keys: list, the list of keys

    Returns:
    value: the value
    """
    if len(keys) == 1:
        return data[keys[0]]
    return extract_nested_value(data[keys[0]], keys[1:])


def training_process_visualizer(
    json_files, metrics, output_file="plot.png", labels=None
):
    """
    Visualize the training data in the .json files

    Args:
    json_files: list, the list of the json files
    metrics: list, the list of the metrics
    output_file: string, the output file
    labels: list, the list of the labels

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    for i, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            data = json.load(f)

        n_train = data["n_train"]

        label = os.path.splitext(os.path.basename(json_file))[0]

        if labels is not None:
            for metric in metrics:
                metric_data = extract_nested_value(data, metric.split("/"))
                plt.plot(
                    n_train, metric_data, label=f"{labels[i]} - {metric.split('/')[-1]}"
                )
                # plt.plot(n_train[20:], metric_data[20:], label=f"{labels[i]} - {metric.split('/')[-1]}")
        else:
            for metric in metrics:
                metric_data = extract_nested_value(data, metric.split("/"))
                plt.plot(
                    n_train, metric_data, label=f"{label} - {metric.split('/')[-1]}"
                )
                # plt.plot(n_train[20:], metric_data[20:], label=f"{label} - {metric.split('/')[-1]}")

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Value")
    plt.title(f"{metric.split('/')[-1]}")
    plt.legend(loc="best")

    # for json_file in json_files:
    #     with open(json_file, "r") as f:
    #         data = json.load(f)

    #     n_train = data["n_train"]
    #     epoch_positions = data["epoch"]

    #     for i, epoch_pos in enumerate(epoch_positions):
    #         plt.axvline(x=n_train[epoch_pos], color='gray', linestyle='--', linewidth=0.5)
    #         plt.text(n_train[epoch_pos], plt.ylim()[0], f'Epoch {i+1}', rotation=90, verticalalignment='bottom')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
