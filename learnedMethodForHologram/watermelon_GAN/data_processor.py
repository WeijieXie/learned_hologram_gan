import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from ..utilities import try_gpu


class data_loader_amp_phs_img_depth(Dataset):

    def __init__(
        self,
        img_path,
        depth_path,
        amp_path,
        phs_path,
        samplesNum=3800,
        channlesNum=3,
        height=192,
        width=192,
        cuda=False,
    ):
        self.dataShape = (samplesNum, channlesNum, height, width)
        self.img = np.memmap(img_path, dtype=np.float32, mode="r", shape=self.dataShape)
        self.depth = np.memmap(
            depth_path, dtype=np.float32, mode="r", shape=self.dataShape
        )
        self.amp = np.memmap(amp_path, dtype=np.float32, mode="r", shape=self.dataShape)
        self.phs = np.memmap(phs_path, dtype=np.float32, mode="r", shape=self.dataShape)

        if cuda:
            self.device = try_gpu()
        else:
            self.device = torch.device("cpu")

    def __len__(self):
        return self.dataShape[0]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        return (
            torch.cat(
                (
                    torch.tensor(self.img[idx]).to(self.device),
                    torch.tensor(self.depth[idx][0]).to(self.device).unsqueeze(0),
                ),
                dim=0,
            ),
            torch.tensor(self.amp[idx]).to(self.device),
            torch.tensor(self.phs[idx]).to(self.device),
        )


class data_loader_amp_2PIphs(Dataset):
    def __init__(
        self,
        amp_path,
        phs_path,
        samplesNum=3800,
        channlesNum=3,
        height=192,
        width=192,
        cuda=False,
    ):
        self.dataShape = (samplesNum, channlesNum, height, width)
        self.amp = np.memmap(amp_path, dtype=np.float32, mode="r", shape=self.dataShape)
        self.phs = np.memmap(phs_path, dtype=np.float32, mode="r", shape=self.dataShape)
        if cuda:
            self.device = try_gpu()
        else:
            self.device = torch.device("cpu")

    def __len__(self):
        return self.dataShape[0]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        return (
            torch.tensor(self.amp[idx]).to(self.device),
            2 * torch.pi * (torch.tensor(self.phs[idx]).to(self.device)),
        )
