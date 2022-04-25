import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from visualization.inverse_normalize import inverse_normalize

def show_dataset(dataset, num_imgs=4, mean=None, std=None, label_classes=None, RANDOM_SEED=42):
    """ Show images from a PyTorch dataset.

    Args:
        dataset (_type_): _description_
        num_imgs (int, optional): _description_. Defaults to 4.
        mean (_type_, optional): _description_. Defaults to None.
        std (_type_, optional): _description_. Defaults to None.
        label_classes (_type_, optional): _description_. Defaults to None.
        RANDOM_SEED (int, optional): _description_. Defaults to 42.
    """
    np.random.seed(RANDOM_SEED)
    random_indices = np.random.randint(low=0, high=len(dataset), size=num_imgs)
    images_tensor = [dataset[i][0] for i in random_indices]
    images_grid = make_grid(tensor=images_tensor, nrow=num_imgs)
    labels = [dataset[i][1] for i in random_indices]
    
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
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)
