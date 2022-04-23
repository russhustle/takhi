import torch

def inverse_normalize(image_tensor, mean, std):
    """ Un-normalization
    Args:
        image_tensor (tensor): image tensor after normalization.
        mean (tuple): means of each channel before normalization.
        std (tuple): standard deviations of each channel before normalization.
    Returns:
        image_tensor (tensor): image tensor before normalization
    """
    mean = torch.as_tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)
    std = torch.as_tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image_tensor.mul_(std).add_(mean)
    return image_tensor
