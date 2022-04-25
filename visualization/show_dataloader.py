import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_dataloader(dataloader, mean=None, std=None, num_imgs=4, label_classes=None):
    """ Show sample images in a PyTorch dataloader.
    Args:
        dataloader (Dataloader): PyTorch dataloader to show.
        mean (List, optional): Mean value for each channel. Defaults to None.
        std (List, optional): Standard deviation for each channel. Defaults to None.
        num_imgs (int, optional): Number of images to show. Defaults to 4.
        label_classes (Dict, optional): Class dictionary. Defaults to None.
    """
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    labels = list(labels[:num_imgs].numpy())
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    images_grid = make_grid(images[:num_imgs])
    
    # Un-normalize
    if mean is not None:
        mean = torch.as_tensor(mean, dtype=images_grid.dtype, device=images_grid.device)
        std = torch.as_tensor(std, dtype=images_grid.dtype, device=images_grid.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        images_grid.mul_(std).add_(mean)
    
    images_numpy = images_grid.numpy()
    images_transposed_numpy = np.transpose(images_numpy, axes=(1,2,0))
    plt.figure(figsize=(num_imgs*5, num_imgs*3))
    plt.axis("off")
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)