import torch


def calculate_psnr(hat, ground_truth):
    assert hat.shape == ground_truth.shape == (3, *hat.shape[1:])

    mse = torch.mean((hat - ground_truth) ** 2, dim=(-1, -2))

    max_vals_dim1, _ = torch.max(ground_truth, dim=-2)
    max_vals, _ = torch.max(max_vals_dim1, dim=-1)

    psnr_per_channel = 10 * torch.log10(max_vals**2 / mse)

    psnr = psnr_per_channel.mean().item()

    return (
        psnr_per_channel[0].item(),
        psnr_per_channel[1].item(),
        psnr_per_channel[2].item(),
        psnr,
    )
