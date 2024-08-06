import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms

from ..utilities import try_gpu


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        feature_map_layers=[3, 8, 13, 22, 31],
        cuda=True,
    ):
        super(PerceptualLoss, self).__init__()
        self.device = try_gpu() if cuda else torch.device("cpu")

        # pick feature maps from the following layers
        self.feature_map_layers = feature_map_layers
        self.feature_map_layers_num = len(self.feature_map_layers)

        vgg19 = (
            torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)
            .features.to(self.device)
            .eval()
        )

        self.net = nn.Sequential(
            *list(vgg19.children())[: max(self.feature_map_layers) + 1]
        ).to(self.device)

        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, hat, target):
        loss = torch.zeros(1, device=self.device)

        x = torch.cat((hat, target), dim=0)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            x
        )

        for name, layer in self.net._modules.items():
            x = layer(x)
            if int(name) in self.feature_map_layers:
                loss += F.mse_loss(x[: hat.size(0)], x[hat.size(0) :])

        return loss / self.feature_map_layers_num


def total_variation(tensor):
    """
    Compute the total variation of a tensor.
    """
    # Compute the difference between the adjacent pixels
    diff1 = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    diff2 = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

    # Compute the total variation
    loss = torch.mean(torch.abs(diff1)) + torch.mean(torch.abs(diff2))

    return loss


def total_variation_for_POH(tensor):
    """
    designed for POH
    """

    diff1 = tensor[:, :, :, 2:] - tensor[:, :, :, :-2]
    diff2 = tensor[:, :, 2:, :] - tensor[:, :, :-2, :]

    # Compute the total variation
    loss = torch.mean(torch.abs(diff1)) + torch.mean(torch.abs(diff2))

    return loss


def total_variation_loss(y_hat, y):
    """
    Compute the total variation loss.
    """
    return torch.abs(total_variation(y_hat) - total_variation(y))


def amp_loss(amp_hat, amp, alpha=1.0):
    loss_l2 = F.mse_loss(amp_hat, amp)
    loss_tv = total_variation_loss(amp_hat, amp)
    return loss_l2 + alpha * loss_tv


def amp_phs_loss(amp_hat, phs_hat, amp, phs, alpha=1.0):
    """
    The input phase is in the range of [0, 2*pi].
    """

    amp_sincos_phs_hat = torch.cat(
        (amp_hat, torch.sin(phs_hat), torch.cos(phs_hat)), dim=1
    )
    amp_sincos_phs = torch.cat((amp, torch.sin(phs), torch.cos(phs)), dim=1)

    loss_l2 = F.mse_loss(amp_sincos_phs_hat, amp_sincos_phs)
    loss_tv = total_variation_loss(amp_sincos_phs_hat, amp_sincos_phs)

    return loss_l2 + alpha * loss_tv


def focal_freq_loss(fake_freq, real_freq):
    frequency_diff = torch.abs(fake_freq - real_freq)

    with torch.no_grad():
        weight_matrix = torch.pow(frequency_diff, 1)  # α = 1
        weight_matrix = weight_matrix / torch.max(weight_matrix)

    weighted_diff = frequency_diff * weight_matrix
    freq_loss = torch.mean(weighted_diff**2)
    return freq_loss
