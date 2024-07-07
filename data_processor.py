import numpy as np
import torch
import os
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


def get_files_in_dir(directory):
    fileNames = os.listdir(directory)
    filePaths = [os.path.join(directory, name) for name in fileNames]
    return filePaths


def read_exr(filename, plot=False):
    exr_file = OpenEXR.InputFile(filename)

    header = exr_file.header()
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
    img = np.stack([R, G, B], axis=-1)

    if plot:
        plt.imshow(img)
        plt.show()

    transform = transforms.Compose([transforms.ToTensor()])

    return transform(img)


class data_generator:
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

        self.output = torch.zeros(self.samplesNum, channelsNum * height * width)

    def __iter__(self):
        return self.read_exrs()

    # def __len__(self):
    #     return len(self.filePaths)

    # def __getitem__(self, idx):
    #     if idx < 0 or idx >= len(self.filePaths):
    #         raise IndexError("Index out of range")
    #     return read_exr(self.filePaths[idx])

    def read_exrs(self):
        for filePath in self.filePaths:
            yield read_exr(filePath)

    def save_as_torch(self):
        for i, tensor in enumerate(self):
            self.output[i] = torch.flatten(tensor)
        torch.save(self.output, os.path.join(self.des, self.folderName + ".pt"))


def read_exr_in_multi_folders(directory, channlesNum=3, height=192, width=192):
    """
    Read exr files in multiple folders and save them as torch tensors
    """
    # only read folders in the directory
    folders = [
        folder
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]
    for folder in folders:
        generator = data_generator(
            os.path.join(directory, folder),
            channelsNum=channlesNum,
            height=height,
            width=width,
        )
        generator.save_as_torch()
        print(f"Saved {os.path.join(directory, folder)}.pt")


class data_loader(Dataset):

    def __init__(
        self,
        amp_path,
        phs_path,
        img_path,
        depth_path,
        channlesNum=3,
        height=192,
        width=192,
    ):
        self.amp = torch.load(amp_path).view(-1, channlesNum, height, width)
        self.phs = torch.load(phs_path).view(-1, channlesNum, height, width)
        self.img = torch.load(img_path).view(-1, channlesNum, height, width)
        self.depth = torch.load(depth_path).view(-1, channlesNum, height, width)

        self.channelsNum = channlesNum
        self.height = height
        self.width = width

    def __len__(self):
        return self.amp.size(0)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.amp.size(0):
            raise IndexError("Index out of range")
        return self.amp[idx], self.phs[idx], self.img[idx], self.depth[idx]
