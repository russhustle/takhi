from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
import torch
import os

def cifar10_dataloaders(BATCH_SIZE=32):
    """ Load CIFAR10 dataset as train, validation and test dataloaders.
    Args:
        BATCH_SIZE (int, optional): Batch size. Defaults to 32.
    Returns:
        train_dataloader
        val_dataloader
        test_dataloader
    """
    # Transoform
    mean=[0.49139968, 0.48215841, 0.44653091]
    std =[0.24703223, 0.24348513, 0.26158784]
    
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    val_test_transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    
    # Datasets
    train_dataset = CIFAR10(root=os.getcwd(), train=True, download=True, transform=train_transform)
    val_length = 5000
    train_length = len(train_dataset)-val_length
    lengths=[train_length, val_length]
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(dataset=train_dataset, lengths=lengths, generator=generator)

    val_dataset = CIFAR10(root=os.getcwd(), train=True, download=True, transform=val_test_transform)
    _, val_dataset = random_split(dataset=val_dataset, lengths=lengths, generator=generator)

    test_dataset = CIFAR10(root=os.getcwd(), train=False, download=False, transform=val_test_transform)
    
    # Dataloaders
    num_workers = int(os.cpu_count()/2)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    
    return train_dataloader, val_dataloader, test_dataloader
