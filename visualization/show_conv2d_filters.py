import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_conv2d_filters(weights):
    """ Show the filters of Conv2D.
    Args:
        weights : Weights of a Conv2D layer.
    Example:
        from torchvision import models
        resnet18_pretrained = models.resnet18(pretrained=True)
        weight = next(resnet18_pretrained.parameters()).data.cpu()
        show_conv2d_filters(weight)
    """
    # Normalize
    min_weight = torch.min(weights)
    weights_normalized = (-1/(2*min_weight))*weights + 0.5
    # Make grid
    grid_size=len(weights_normalized)
    weights_list = [weights_normalized[i] for i in range(grid_size)]
    weights_grid = make_grid(tensor=weights_list, nrow=8, padding=1)
    print(f"Weight grid shape: {weights_grid.shape}")
    # Visualization
    plt.figure(figsize=(10,10))
    weight_grid_numpy = weights_grid.numpy()
    weight_grid_numpy_transposed = np.transpose(weight_grid_numpy, (1,2,0))
    plt.axis("off")
    plt.imshow(weight_grid_numpy_transposed)
