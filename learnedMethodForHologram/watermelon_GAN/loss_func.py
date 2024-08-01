import torch
from torch.nn import functional as F


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


def total_variation_loss(y_hat, y):
    """
    Compute the total variation loss.
    """
    return total_variation(y_hat) - total_variation(y)


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
