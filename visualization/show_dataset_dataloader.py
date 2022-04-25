import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_dataset_dataloader(data, datatype, mean=None, std=None, num_imgs=4, label_classes=None):
    """_summary_

    Args:
        data (_type_): _description_
        datatype (_type_): _description_
        mean (_type_, optional): _description_. Defaults to None.
        std (_type_, optional): _description_. Defaults to None.
        num_imgs (int, optional): _description_. Defaults to 4.
        label_classes (_type_, optional): _description_. Defaults to None.
    """
    # Make grid
    if datatype == "dataloader":
        dataiter = iter(data)
        images, labels = dataiter.next()
        labels = list(labels[:num_imgs].numpy())
        images_grid = make_grid(images[:num_imgs])
    
    if datatype == "dataset":
        random_indices = np.random.randint(low=0, high=len(data), size=num_imgs)
        images_tensor = [data[i][0] for i in random_indices]
        labels = [data[i][1] for i in random_indices]
        images_grid = make_grid(tensor=images_tensor, nrow=num_imgs)

    # Un-normalize
    if mean is not None:
        mean = torch.as_tensor(mean, dtype=images_grid.dtype, device=images_grid.device)
        std = torch.as_tensor(std, dtype=images_grid.dtype, device=images_grid.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        images_grid.mul_(std).add_(mean)
    
    # Visualization
    images_numpy = images_grid.numpy()
    images_transposed_numpy = np.transpose(images_numpy, axes=(1,2,0))
    plt.figure(figsize=(num_imgs*5, num_imgs*3))
    plt.axis("off")
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)