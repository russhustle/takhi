import numpy as np
import matplotlib.pyplot as plt

def show_tensor(tensor):
    """
    Show tensor image or image grids.
    """
    array = tensor.numpy()
    array_transposed = np.transpose(a=array, axes=(1,2,0))
    plt.axis("off")
    plt.imshow(array_transposed)
