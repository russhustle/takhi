Visualization for Computer Vision
===

1. Show PyTorch dataloaders
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

2. `show_dataloader` [[link](https://github.com/Sihan-A/takhi/blob/main/cv/viz.py)]