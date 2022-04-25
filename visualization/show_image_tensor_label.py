import numpy as np
import torch
import matplotlib.pyplot as plt
from visualization.inverse_normalize import inverse_normalize

def show_image_tensor_label(image_tensor, label, label_classes=None, mean=None, std=None):
    """_summary_

    Args:
        image_tensor (_type_): _description_
        label (_type_): _description_
        label_classes (_type_, optional): _description_. Defaults to None.
        mean (_type_, optional): _description_. Defaults to None.
        std (_type_, optional): _description_. Defaults to None.
    """
    
    if mean is not None:
        mean = torch.as_tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)
        std = torch.as_tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image_tensor.mul_(std).add_(mean)
    
    image_numpy = image_tensor.numpy()
    image_numpy_transposed = np.transpose(a=image_numpy, axes=(1,2,0))
    plt.title(f"Label: {label}" if label_classes is None else label_classes[label])
    plt.axis("off")
    plt.imshow(image_numpy_transposed)
