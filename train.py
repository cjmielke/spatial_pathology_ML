from datetime import timedelta

import pytorch_lightning as pl
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models import Model

parser = argparse.ArgumentParser()
parser.add_argument('--timm', default='resnet50', help='TIMM encoder to use')
parser.add_argument('--dropout', default=0.1)
args = parser.parse_args()

from hest_datasets import dataset

model = Model(args, dataset.num_genes)

cfg = model.encoder.default_cfg
dataset.make_transform(mean=cfg['mean'], std=cfg['std'])

logger = WandbLogger(project='st', log_model='all')
logger.log_hyperparams(dict(
    num_tiles=dataset.num_tiles,
    num_slides=dataset.num_slides,
    num_genes=dataset.num_genes
))

train_loader = DataLoader(dataset, num_workers=1, shuffle=True, batch_size=8, drop_last=True)

callbacks = [
    ModelCheckpoint('./checkpoints', save_top_k=2, monitor="train_loss", mode="min",
                    train_time_interval=timedelta(minutes=60))
]


if __name__ == '__main__':

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=32,
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloaders=train_loader)
