import numpy as np
import matplotlib.pyplot as plt
from visualization.inverse_normalize import inverse_normalize

def show_image_tensor_label(image_tensor, label, label_classes=None, mean=None, std=None):
    if mean is not None:
        images_grid = inverse_normalize(image_tensor, mean, std) 
    image_numpy = image_tensor.numpy()
    image_numpy_transposed = np.transpose(a=image_numpy, axes=(1,2,0))
    plt.title(f"Label: {label}" if label_classes is None else label_classes[label])
    plt.axis("off")
    plt.imshow(image_numpy_transposed)
