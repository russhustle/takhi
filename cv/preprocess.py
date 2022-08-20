import torch
import numpy as np
    
"""
- mean_std_dataloader: calculate mean and standard deviation of a dataloader.
- mean_std_dataset: calculate mean and standard deviation of a dataset.
"""

def mean_std_dataloader(dataloader):
    """ Calculate mean and standard deviation of a PyTorch dataloader.
    Args:
        dataloader: PyTorch DataLoader.
    Returns:
       mean, std: mean and standard deviation.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

def mean_std_dataset(dataset):
    """ Calculate mean and standard deviation of a PyTorch dataset.
    Args:
        dataloader: PyTorch Dataset.
    Returns:
       mean, std: mean and standard deviation.
    """
    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    mean = [meanR, meanG, meanB]
    std = [stdR, stdG, stdB]
    return mean, std
