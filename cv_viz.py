import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

"""
- inverse_normalize
- show_dataset_dataloader
- make_grid_dataloader
- make_grid_dataset
- show_tensor
"""

def inverse_normalize(image_tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)
    std = torch.as_tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image_tensor.mul_(std).add_(mean)
    return image_tensor

def show_dataset_dataloader(data, datatype="dataloader", mean=None, std=None, num_imgs=4, label_classes=None):
    if datatype == "dataloader":
        images_grid, labels = make_grid_dataloader(data, num_imgs, label_classes)
    if datatype == "dataset":
        images_grid, labels = make_grid_dataset(data, num_imgs, label_classes)
    if mean is not None:
        images_grid = inverse_normalize(images_grid, mean, std)
    images_numpy = images_grid.numpy()
    images_transposed_numpy = np.transpose(images_numpy, axes=(1,2,0))
    plt.figure(figsize=(num_imgs*5, num_imgs*3))
    plt.axis("off")
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)

def show_dataloader(dataloader, mean=None, std=None, num_imgs=4, label_classes=None):
    images_grid, labels = make_grid_dataloader(dataloader, num_imgs, label_classes)
    if mean is not None:
        images_grid = inverse_normalize(images_grid, mean, std)
    images_numpy = images_grid.numpy()
    images_transposed_numpy = np.transpose(images_numpy, axes=(1,2,0))
    plt.figure(figsize=(num_imgs*5, num_imgs*3))
    plt.axis("off")
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)

def show_dataset(dataset, num_imgs=4, mean=None, std=None, label_classes=None):
    images_grid, labels = make_grid_dataset(dataset, num_imgs=num_imgs, label_classes=label_classes)
    if mean is not None:
        images_grid = inverse_normalize(image_tensor=images_grid, mean=mean, std=std)
    images_numpy = images_grid.numpy()
    images_transposed_numpy = np.transpose(images_numpy, axes=(1,2,0))
    plt.figure(figsize=(num_imgs*5, num_imgs*3))
    plt.axis("off")
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)

def make_grid_dataloader(dataloader, num_imgs=4, label_classes=None):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    labels = list(labels[:num_imgs].numpy())
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    images_grid = make_grid(images[:num_imgs])
    return images_grid, labels

def make_grid_dataset(dataset, num_imgs=4, label_classes=None, RANDOM_SEED=42):
    np.random.seed(RANDOM_SEED)
    random_indices = np.random.randint(low=0, high=len(dataset), size=num_imgs)
    images_tensor = [dataset[i][0] for i in random_indices]
    labels = [dataset[i][1] for i in random_indices]
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    images_grid = make_grid(tensor=images_tensor, nrow=dataset)
    return images_grid, labels

def show_tensor(tensor):
    array = tensor.numpy()
    array_transposed = np.transpose(a=array, axes=(1,2,0))
    plt.axis("off")
    plt.imshow(array_transposed)





