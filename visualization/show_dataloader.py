import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from visualization.inverse_normalize import inverse_normalize

def show_dataloader(dataloader, mean=None, std=None, num_imgs=4, label_classes=None):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    labels = list(labels[:num_imgs].numpy())
    if label_classes is not None:
        labels = [label_classes[i] for i in labels]
    images_grid = make_grid(images[:num_imgs])
    if mean is not None:
        images_grid = inverse_normalize(image_tensor=images_grid, mean=mean, std=std)
    images_numpy = images_grid.numpy()
    images_transposed_numpy = np.transpose(images_numpy, axes=(1,2,0))
    plt.figure(figsize=(num_imgs*5, num_imgs*3))
    plt.axis("off")
    plt.title(f"Label: {labels}")
    plt.imshow(images_transposed_numpy)
    