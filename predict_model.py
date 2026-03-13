"""Script for training RainNet using PyTorch Lightning API."""
from pathlib import Path
import argparse
import random
import numpy as np

from utils.config import load_config

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from datamodules.datamoduleSVKDualPol import SlovakDataModuleDualPol

from models import RainNet3D_OneShot as RN
from models import RainNet3D_OneShot_GAN as RN_GAN
from models import MF3DUNet as MF3DUNet

from utils import HDF5Writer

import datetime


def main(config, checkpoint=None):
    confpath = Path("config") / config
    dsconf = load_config(confpath / "datasets.yaml")
    modelconf = load_config(confpath / "model.yaml")

    if checkpoint is None:
        raise ValueError("Checkpoint path must be provided. Use the -c or --checkpoint argument to specify the path.")

    datamodel = SlovakDataModuleDualPol(dsconf, modelconf.train_params)

    if modelconf.architecture == "RainNet3Dconv_2Ddata":
        model = RN.load_from_checkpoint(checkpoint, config=modelconf, map_location=torch.device('cpu'))
    elif modelconf.architecture == "RainNet3Dconv_2Ddata_GAN":
        model = RN_GAN.load_from_checkpoint(checkpoint, config=modelconf, map_location=torch.device('cpu'))
    elif modelconf.architecture == "MF-3D-U-Net":
        model = MF3DUNet.load_from_checkpoint(checkpoint, config=modelconf, map_location=torch.device('cpu'))
    else:
        raise NotImplementedError(f"Architecture {config.architecture} not implemented!")
    
    output_writer = HDF5Writer(**modelconf.prediction_output)

    trainer = pl.Trainer(
        devices=modelconf.train_params.gpus,
        callbacks=[output_writer],
    )

    trainer.predict(model, datamodel, return_predictions=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for model.",
    )
    args = argparser.parse_args()
    main(args.config, args.checkpoint)
