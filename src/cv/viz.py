import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

"""
- inverse_normalize
- make_grid_dataset
- make_grid_dataloader
- show_dataset
- show_dataloader
- show_tensor
- def show_image_tensor_label
- show_conv2d_filters
"""


def show_dataloader(
    dataloader, mean=None, std=None, num_imgs=4, label_classes=None
):
    """
    from takhi.cv.viz import show_dataloader
    show_dataloader(dataloader, mean=None, std=None, num_imgs=4, label_classes=None)
    """
    images_grid, labels = make_grid_dataloader(
        dataloader, num_imgs, label_classes
    )
    if mean is not None:
        images_grid = inverse_normalize(images_grid, mean, std)
    plt.figure(figsize=(num_imgs * 5, num_imgs * 3))
    plt.title(f"Labels: {labels}")
    show_tensor(images_grid)


def inverse_normalize(image_tensor, mean, std):
    mean = torch.as_tensor(
        mean, dtype=image_tensor.dtype, device=image_tensor.device
    )
    std = torch.as_tensor(
        std, dtype=image_tensor.dtype, device=image_tensor.device
    )
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image_tensor.mul_(std).add_(mean)
    return image_tensor


def make_grid_dataset(dataset, num_imgs=4, label_classes=None, RANDOM_SEED=42):
    np.random.seed(RANDOM_SEED)
    random_indices = np.random.randint(low=0, high=len(dataset), size=num_imgs)
    images_tensor = [dataset[i][0] for i in random_indices]
    labels = [dataset[i][1] for i in random_indices]
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    images_grid = make_grid(tensor=images_tensor, nrow=num_imgs)
    return images_grid, labels


def make_grid_dataloader(dataloader, num_imgs=4, label_classes=None):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    labels = list(labels[:num_imgs].numpy())
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    images_grid = make_grid(images[:num_imgs])
    return images_grid, labels


def show_dataset(dataset, num_imgs=4, mean=None, std=None, label_classes=None):
    images_grid, labels = make_grid_dataset(
        dataset, num_imgs=num_imgs, label_classes=label_classes
    )
    if mean is not None:
        images_grid = inverse_normalize(
            image_tensor=images_grid, mean=mean, std=std
        )
    plt.figure(figsize=(num_imgs * 5, num_imgs * 3))
    plt.title(f"Labels: {labels}")
    show_tensor(images_grid)


def show_tensor(tensor):
    array = tensor.numpy()
    array_transposed = np.transpose(a=array, axes=(1, 2, 0))
    plt.axis("off")
    plt.imshow(array_transposed)


def show_image_tensor_label(
    image_tensor, label, label_classes=None, mean=None, std=None
):
    # Show an iamge tensor with label
    if mean is not None:
        image_tensor = inverse_normalize(image_tensor, mean=mean, std=std)
    plt.title(
        f"Label: {label}" if label_classes is None else label_classes[label]
    )
    show_tensor(image_tensor)


def show_conv2d_filters(weights):
    # Normalize
    min_weight = torch.min(weights)
    weights_normalized = (-1 / (2 * min_weight)) * weights + 0.5
    # Make grid
    grid_size = len(weights_normalized)
    weights_list = [weights_normalized[i] for i in range(grid_size)]
    weights_grid = make_grid(tensor=weights_list, nrow=8, padding=1)
    print(f"Weight grid shape: {weights_grid.shape}")
    # Visualization
    plt.figure(figsize=(10, 10))
    show_tensor(weights_grid)
