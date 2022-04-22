import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from cv.normalization import inverse_normalize

def show_dataloader(dataloader, mean=None, std=None, num_imgs=4):
    """ Show the PyTorch dataloader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader
        mean (list, optional): Mean value used in normalization. Defaults to None.
        std (list, optional): Standard deviation used in normalization. Defaults to None.
        num_imgs (int, optional): Number of images to show. Defaults to 4.
    """

    plt.figure(figsize=(10,10))
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    images = make_grid(images[:num_imgs])
    labels = list(labels[:num_imgs].numpy())
    if mean is not None:
        images = inverse_normalize(image_tensor=images, mean=mean, std=std)
    np_images = images.numpy()
    plt.title(f"{labels}")
    plt.imshow(np.transpose(np_images, (1,2,0)))
