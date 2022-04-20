import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from cv.normalization import inverse_normalize

def imageshow(image):
    """_summary_
    Args:
        image (_type_): _description_
    """
    # unnormalize
    image = inverse_normalize(image, mean=(0.4914, 0.4822, 0.4465),std=(0.2470, 0.2435, 0.2616))
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1,2,0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()
num_images = 4
imageshow(make_grid(images[:4]))
print(' '+' || '.join(classes[labels[j]] for j in range(num_images)))

#####################################################################

def imageshow(img, mean, std, text=None):
    """ Show img
    Args:
        img (_type_): _description_
        mean (_type_): _description_
        std (_type_): _description_
        text (_type_, optional): _description_. Defaults to None.
    """
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if text is not None:
        plt.title(text)

imgs, cls = next(iter(train_loader))
imageshow(
    make_grid(imgs),
    mean=(0.5143, 0.4760, 0.3487),
    std=(0.2814, 0.2625, 0.2912),
    text=[classes[c] for c in cls],
)