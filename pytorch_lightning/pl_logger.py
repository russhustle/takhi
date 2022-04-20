from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("lightning_logs/", name="model")

"""
%reload_ext tensorboard
%tensorboard --logdir lightning_logs/
"""