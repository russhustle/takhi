import os
from PIL import Image


def reshape_image(image, shape):
    """
    Resize an image to the given shape.
    Args:
        image (_type_): _description_
        shape (_type_): _description_
    Returns:
        _type_: _description_
    """
    return image.resize(shape, Image.ANTIALIAS)


def reshape_images(image_path, output_path, shape):
    """
    Reshape the images in 'image_path' and save into 'output_path'.

    Args:
        image_path (_type_): _description_
        output_path (_type_): _description_
        shape (list): _description_
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = os.listdir(image_path)
    num_image = len(images)

    for i, image in enumerate(images):
        with open(os.path.join(image_path, image), "r+b") as f:
            with Image.open(f) as image:
                image = reshape_image(image, shape)
                image.save(os.path.join(output_path, image), image.format)
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'.".format(i + 1, num_image, output_path))


image_path = "./data_dir/train2014/"
output_path = "./data_dir/resized_images/"
image_shape = [256, 256]

reshape_images(image_path, output_path, image_shape)
