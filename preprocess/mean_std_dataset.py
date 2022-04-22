import numpy as np

def mean_std_dataset(dataset):
    """ Return mean and standard deviation of a PyTorch dataset.
    Args:
        dataset (Dataset): PyTorch dataset.
    Returns:
        mean: Mean value for each channel.
        std: Standard deviation for each channel.
    """
    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    mean = [meanR, meanG, meanB]
    std = [stdR, stdG, stdB]
    return mean, std
