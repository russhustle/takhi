Datasets
===

CIFAR10
---

```python
cifar10_label_classes = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}
cifar10_mean=[0.49139968, 0.48215841, 0.44653091]
cifar10_std =[0.24703223, 0.24348513, 0.26158784]
```

```python
from takhi.dataset.cifar10 import cifar10_dataloaders
train_dataloader, val_dataloader, test_dataloader = cifar10_dataloaders()
```

