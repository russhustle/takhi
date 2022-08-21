Visualization for Computer Vision
===

1. Show PyTorch dataloaders

Show PyTorch dataloaders
---

1. Weights & Biases [[link](https://docs.wandb.ai/ref/python/data-types/image)]

```python
import wandb
import random
xs, ys = next(iter(dataloader))
idx = random.randint(0, len(xs) - 1)
print(f"Label: {classes[int(ys[idx])]}")
wandb.Image(xs[idx]).image
```

This will show one sample, if we want to show several images at the same time, we should make a tensor grid first (`make_grid_dataloader` [[link](https://github.com/Sihan-A/takhi/blob/main/cv/viz.py)]).

2. `show_dataloader` [[link](https://github.com/Sihan-A/takhi/blob/main/cv/viz.py)]

