from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
import torch
from pytorch_lightning import LightningDataModule
import os

cifar10_mean=[0.49139968, 0.48215841, 0.44653091]
cifar10_std =[0.24703223, 0.24348513, 0.26158784]
num_workers = int(os.cpu_count()/2)

def cifar10_dataloaders(BATCH_SIZE=32):
    """ Load CIFAR10 dataset as train, validation and test dataloaders.
    Args:
        BATCH_SIZE (int, optional): Batch size. Defaults to 32.
    Returns:
        train_dataloader
        val_dataloader
        test_dataloader
    """
    train_transform = Compose([
        RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
        Normalize(mean=cifar10_mean, std=cifar10_std),
        ])
    val_test_transform = Compose([
        ToTensor(), Normalize(mean=cifar10_mean, std=cifar10_std),
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
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, drop_last=True, pin_memory=True,)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, drop_last=True, pin_memory=True,)
    
    return train_dataloader, val_dataloader, test_dataloader

class CIFAR10DataModule(LightningDataModule):
    
    def __init__(self, data_dir: str=None, val_split: int=5000,
                 num_workers: int=2, batch_size: int=32, seed: int=42,):
        super().__init__(self)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split
        self.seed = seed
    
    def prepare_data(self):
        self.DATASET(self.data_dir, train=True, download=True, transform=ToTensor(),)
        self.DATASET(self.data_dir, train=False, download=True, transform=ToTensor(),)
    
    def train_dataloader(self):
        transforms = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(),
            Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std =[0.24703223, 0.24348513, 0.26158784]),])
        dataset = self.DATASET(root=self.data_dir, train=True, download=False, transform=transforms,)
        train_length = len(dataset)
        dataset_train, _ = random_split(dataset=dataset, lengths=[train_length-self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed))
        loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, pin_memory=True)
        return loader

    def val_dataloader(self):
        transforms = Compose([ToTensor(), Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std =[0.24703223, 0.24348513, 0.26158784]),])
        dataset = self.DATASET(root=self.data_dir, train=True, download=False, transform=transforms,)
        train_length = len(dataset)
        _, dataset_val = random_split(dataset=dataset, lengths=[train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed),)
        loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)
        return loader

    def test_dataloader(self):
        transforms = Compose([ToTensor(), Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std =[0.24703223, 0.24348513, 0.26158784]),])
        dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms,)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, pin_memory=True,)
        return loader
    