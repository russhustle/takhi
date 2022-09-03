PyTorch Lightning
===

[TOC]

Callbacks
---

```python
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath='checkpoint',
    filename='model-{epoch:02d}-{val_loss:.2f}',
)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar
progress_bar = LitProgressBar(refresh_rate=10, process_position=0)
callbacks = [checkpoint_callback, progress_bar, early_stopping, lr_monitor]
```

Loggers
---

### TensorBoard

```python
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("lightning_logs/", name="model")
```

```python
%reload_ext tensorboard
%tensorboard --logdir lightning_logs/
```

### Weights & Biases

Trainer
---

```python
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import pl_callbacks

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

model = LitResnet(lr=0.05)
datamodule = CIFAR10DataModule(data_dir="./", batch_size=BATCH_SIZE)
callbacks = pl_callbacks()
trainer = Trainer(max_epochs=100, gpus=AVAIL_GPUS, logger=logger,callbacks=callbacks,)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
```

