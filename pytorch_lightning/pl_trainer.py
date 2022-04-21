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
trainer = Trainer(
    max_epochs=100,
    gpus=AVAIL_GPUS,
    logger=logger,
    callbacks=callbacks,
)
trainer.fit(model, datamodule)
trainer.test(model, datamodule=datamodule)