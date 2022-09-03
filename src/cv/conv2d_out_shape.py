import numpy as np


def findConv2dOutShape(H_in, W_in, conv, pool=2):
    """Find out the shape of Conv2D output shape

    Args:
        H_in (int): _description_
        W_in (int): _description_
        conv (Conv2D): _description_
        pool (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    # Ref: https://pytorch.org/docs/stable/nn.html
    H_out = np.floor(
        (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
        / stride[0]
        + 1
    )
    W_out = np.floor(
        (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
        / stride[1]
        + 1
    )

    if pool:
        H_out /= pool
        W_out /= pool

    return int(H_out), int(W_out)
