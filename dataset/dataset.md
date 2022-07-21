Use dataset
===

```python
%%capture
!pip install pytorch-lightning
!git clone https://github.com/Sihan-A/sihan_utils.git
```

Cifar10
---

```python
from sihan_utils.dataset.CIFAR10 import cifar10_dataloaders
from sihan_utils.dataset.CIFAR10 import cifar10_datasets()
cifar10_mean=[0.49139968, 0.48215841, 0.44653091]
cifar10_std =[0.24703223, 0.24348513, 0.26158784]
cifar10_label_classes={
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

train_dataset, val_dataset, test_dataset = cifar10_datasets()
train_dataloader, val_dataloader, test_dataloader = cifar10_dataloaders()
```

