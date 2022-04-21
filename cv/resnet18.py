from torchvision.models import resnet18
from torch import nn

def create_model(num_classes):
    """ To create a pre-trained resnet18 model 

    Returns:
        model: A pre-trained resnet18 model
    """
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=(3,3),
        stride=(1,1), padding=(1,1), bias=False)
    model.maxpool = nn.Identity()
    return model
