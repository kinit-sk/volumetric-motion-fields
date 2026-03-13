import os

from utils.config import load_config
from pathlib import Path

confpath = Path("/home/ppavlik/repos/nowcasting-SVK-dualpol/config/MF-3D-U-Net_altitude_wise")
dsconf = load_config(confpath / "datasets.yaml")
modelconf = load_config(confpath / "model.yaml")

import datasets.SlovakDatasetDualPol as SlovakDatasetDualPol

ds = SlovakDatasetDualPol(split="full", **dsconf.SHMUDataset)

from models import MF3DUNetAltitudeWise
import torch

checkpoint = Path("/projects/p1364-25-2/checkpoints/Nowcasting_SVK_dualpol/Layers8/epoch=20-step=21000.ckpt")

model = MF3DUNetAltitudeWise.load_from_checkpoint(checkpoint, config=modelconf, map_location=torch.device('cpu'))
model = model.to(torch.device('cuda:0'))

class SpoofTrainer(object):
    pass

spoof_trainer = SpoofTrainer()
spoof_trainer.training = False
spoof_trainer.predicting = False
spoof_trainer.testing = True
spoof_trainer.state = SpoofTrainer()
spoof_trainer.state.stage = 'test'

spoof_trainer.datamodule = SpoofTrainer()
spoof_trainer.datamodule.test_dataset = ds

model._trainer = spoof_trainer
model.eval()

from itertools import combinations_with_replacement, permutations
import torch
import torch.nn as nn
from torchmetrics.regression import PearsonCorrCoef
from torchvision.transforms.functional import center_crop

mse = nn.MSELoss()

def _count_mf_corr(i):
    device = torch.device("cuda:0")
    pearson = PearsonCorrCoef().to(device)

    idx = [i]

    x, y, _ = ds[idx[0]]
    x = center_crop(x, 512).to(device)
    y = center_crop(y, 512).to(device)

    x = x.unsqueeze(0).to(torch.device('cuda:0'))
    y = y.unsqueeze(0).to(torch.device('cuda:0'))

    with torch.no_grad():
        y_hat, mf = model.predict_step((x, y, idx), idx)

    # number of altitude levels
    n_altitudes = x.shape[3]

    pearson_corrs = {}
    mse_mf_losses = {}
    mse_losses = {}

    for z1, z2 in permutations(range(n_altitudes), 2):
        slice1 = x[0, :, 0, z1]
        slice2 = x[0, :, 0, z2]

        sample_mask_1 = slice1.mean(dim=0) > -15
        sample_mask_1 = torch.stack([sample_mask_1, sample_mask_1])

        sample_mask_2 = slice2.mean(dim=0) > -15
        sample_mask_2 = torch.stack([sample_mask_2, sample_mask_2])

        sample_mask = sample_mask_1 | sample_mask_2

        motion_field_1 = mf[z1].squeeze()
        motion_field_2 = mf[z2].squeeze()

        pearson_corr = pearson(motion_field_1[sample_mask].flatten(), motion_field_2[sample_mask].flatten())
        mse_mf_loss = mse(motion_field_1[sample_mask], motion_field_2[sample_mask])
        mse_loss = mse(y_hat.squeeze().nan_to_num(), ds.invScaler(y.squeeze().nan_to_num()))

        pearson_corrs[(z1, z2)] = pearson_corr.cpu().item()
        mse_mf_losses[(z1, z2)] = mse_mf_loss.cpu().item()
        mse_losses[(z1, z2)] = mse_loss.cpu().item()

    return pearson_corrs, mse_mf_losses, mse_losses

from tqdm import tqdm

mf_corrs_list = []
mse_mf_losses_list = []
mse_losses_list = []

for i in tqdm(range(len(ds))):
    mf_corrs, mse_mf_losses, mse_losses = {}, {}, {}
    for retry in range(3):
        try:
            mf_corrs, mse_mf_losses, mse_losses = _count_mf_corr(i)
            break
        except Exception as e:
            if retry == 2:
                print(f"Failed to process sample {i} after 3 retries: {e}")
            else:
                continue
    mf_corrs_list.append(mf_corrs)
    mse_mf_losses_list.append(mse_mf_losses)
    mse_losses_list.append(mse_losses)

import numpy as np

np.savez_compressed("altitude_wise_sample_wise_mf_corrs_full.npz",
                    mf_corrs=mf_corrs_list,
                    mse_mf_losses=mse_mf_losses_list,
                    mse_losses=mse_losses_list)