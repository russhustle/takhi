def get_mean_and_std(dataloader):
    """ Get the mean and standard deviation of a dataloader
    Args:
        dataloader (DataLoader): PyTorch DataLoader
    Returns:
        _type_: _description_
    """
    import torch
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def inverse_normalize(image_tensor, mean, std):
    """ Un-normalization
    Args:
        image_tensor (tensor): image tensor after normalization
        mean (tuple): means of each channel before normalization
        std (tuple): standard deviations of each channel before normalization
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
