import os
from torchvision.datasets import STL10
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader

def stl10_datasets_dataloaders(data="datasets", BATCH_SIZE=32, RANDOM_SEED=42):
    """ Return training, validation and test datasets or dataloaders of STL-10 dataset.
    Args:
        data (str, optional): Dataset of dataloaders. Defaults to "datasets".
        BATCH_SIZE (int, optional): Batch size of dataloaders. Defaults to 32.
        RANDOM_SEED (int, optional): Random seed. Defaults to 42.
    Returns:
        train_dataset, val_dataset, test_dataset
        or
        train_dataloader, val_dataloader, test_dataloader
    """
    train_mean, train_std = [0.4467106, 0.43980986, 0.40664646], [0.22414584, 0.22148906, 0.22389975]
    val_test_mean, val_test_std = [0.44723064, 0.4396425, 0.40495726], [0.22489566, 0.22172786, 0.22369827]
    train_transform = Compose([ToTensor(), Normalize(mean=train_mean, std=train_std),])
    val_test_transform = Compose([ToTensor(), Normalize(mean=val_test_mean, std=val_test_std)])
    
    train_dataset = STL10(root=os.getcwd(), split='train', download=True, transform=train_transform)
    test_dataset_full = STL10(root=os.getcwd(), split='test', download=False, transform=val_test_transform)
    
    indices = list(range(len(test_dataset_full)))
    test_dataset_labels = [y for _, y in test_dataset_full]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    for test_index, val_index in sss.split(indices, test_dataset_labels):
        print("Stratified shuffle sampling...")
    
    val_dataset = Subset(dataset=test_dataset_full, indices=val_index)
    test_dataset = Subset(dataset=test_dataset_full, indices=test_index)
    
    if data == "datasets":
        return train_dataset, val_dataset, test_dataset
    
    if data == "dataloaders":
        num_workers = int(os.cpu_count()//2)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, drop_last=True, pin_memory=True,)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=True, pin_memory=True,)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=True, pin_memory=True,)
        return train_dataloader, val_dataloader, test_dataloader

label_classes = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck",
}
