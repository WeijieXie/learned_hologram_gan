import numpy as np
import torch
import os
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.nn import functional as F

from .utilities import try_gpu


def get_files_in_dir(directory):
    fileNames = os.listdir(directory)
    filePaths = [os.path.join(directory, name) for name in fileNames]
    filePaths.sort()
    return filePaths


def read_exr(filename, plot=False):
    exr_file = OpenEXR.InputFile(filename)

    # header = exr_file.header()
    dw = exr_file.header()["dataWindow"]
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    # print(f"header: {header}")
    # print(f"dataWindow: {dw}")
    # print(f"width: {width}, height: {height}")

    def read_channel(channel):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        str_data = exr_file.channel(channel, pt)
        data = np.frombuffer(str_data, dtype=np.float32)
        data.shape = (height, width)
        return data

    R = read_channel("R")
    G = read_channel("G")
    B = read_channel("B")

    if plot:
        img = np.stack([R, G, B], axis=-1)
        plt.imshow(img)
        plt.show()
        # transform = transforms.Compose([transforms.ToTensor()])

    return np.stack([R, G, B], dtype=np.float32)


class dataConverterExr2Bin:
    """
    A class to read a directory of exr files and save them
    as a torch tensor

    Attributes:
    directory: string, the path of the directory
    des: string, the path of the destination directory
    channelsNum: int, the number of channels
    height: int, the height of the image
    width: int, the width of the image
    """

    def __init__(self, directory, des=None, channelsNum=3, height=192, width=192):
        self.directory = directory
        self.upFolder, self.folderName = os.path.split(directory)
        self.filePaths = get_files_in_dir(directory)
        self.samplesNum = len(self.filePaths)
        self.channelsNum = channelsNum
        self.height = height
        self.width = width

        if des is None:
            self.des = self.upFolder
        else:
            self.des = des

    # def __iter__(self):
    #     return self.read_exrs()

    def __len__(self):
        return len(self.filePaths)

    # def __getitem__(self, idx):
    #     if idx < 0 or idx >= len(self.filePaths):
    #         raise IndexError("Index out of range")
    #     return read_exr(self.filePaths[idx])

    # def read_exrs(self):
    #     for filePath in self.filePaths:
    #         yield read_exr(filePath)

    def save_as_np_array(self):
        output = np.zeros(
            (self.samplesNum, self.channelsNum, self.height, self.width),
            dtype=np.float32,
        )
        # print(f"the shape of the output is {output.shape}")
        for i, filePath in enumerate(self.filePaths):
            sample = read_exr(filePath)
            # print(f"the shape of the sample is {sample.shape}")
            output[i, :, :, :] = sample
        output.tofile(os.path.join(self.des, self.folderName + ".bin"))
        print(
            f"Saved {os.path.join(self.des, self.folderName)}.bin and the size is {os.path.getsize(os.path.join(self.des, self.folderName + '.bin'))}"
        )


def read_exr_in_multi_folders(directory, channlesNum=3, height=192, width=192):
    """
    Read exr files in multiple folders and save them as torch tensors
    """
    # only read folders in the directory without hiding
    folders = [
        folder
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]
    print(f"there are {len(folders)} folders in the directory")
    for folder in folders:
        generator = dataConverterExr2Bin(
            os.path.join(directory, folder),
            channelsNum=channlesNum,
            height=height,
            width=width,
        )
        generator.save_as_np_array()