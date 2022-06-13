from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize)
import torch
import os

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
        transforms = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(
                mean=[0.49139968, 0.48215841, 0.44653091],
                std =[0.24703223, 0.24348513, 0.26158784]),
        ])
        dataset = self.DATASET(
            root=self.data_dir, train=True, download=False, transform=transforms,)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset=dataset,
            lengths=[train_length-self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset=dataset_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, pin_memory=True
        )
        return loader

    def val_dataloader(self):
        transforms = Compose([
            ToTensor(), 
            Normalize(
                mean=[0.49139968, 0.48215841, 0.44653091],
                std =[0.24703223, 0.24348513, 0.26158784]
            ),
        ])
        dataset = self.DATASET(
            root=self.data_dir, train=True, download=False, transform=transforms,
        )
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset=dataset, lengths=[train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed),
        )
        loader = DataLoader(
            dataset=dataset_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        return loader

    def test_dataloader(self):
        transforms = Compose([
            ToTensor(),
            Normalize(
                mean=[0.49139968, 0.48215841, 0.44653091],
                std =[0.24703223, 0.24348513, 0.26158784]),
        ])
        dataset = self.DATASET(
            self.data_dir, train=False, download=False, transform=transforms,)
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, pin_memory=True,
        )
        return loader
    