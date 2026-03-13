# %%
import os
os.chdir("/home/ppavlik/repos/nowcasting-SVK-dualpol")

import datasets.SlovakDatasetDualPol as SlovakDatasetDualPol

ds = SlovakDatasetDualPol(path=fr"/home/projects/p1364-25-2/datasets/SHMU_4_New",
                   metadata_folder=fr"/home/projects/p1364-25-2/datasets/SHMU_4_New/metadata",
                   split="full", denoise=True, erosion_mask=False, transform_to_mmh=False,
                   input_block_length=1, prediction_block_length=0,
                   products=['dBZ', 'RhoHV'], aggregate_vertical=False)

# %%
from itertools import combinations
import torch
from tqdm import tqdm
from torchmetrics.regression import PearsonCorrCoef
from torchvision.transforms.functional import center_crop
import numpy as np

def _count_precip_ratios(i):
    idx = [i]

    x, _, _ = ds[idx[0]]

    # number of altitude levels
    n_altitudes = x.shape[2]

    results = {}

    for z in range(n_altitudes):
        p = x[0, 0, z]

        ratio_mask_0 = p > 0
        ratio_mask_20 = p > 20
        ratio_mask_40 = p > 40

        pixels = x.shape[-2] * x.shape[-1]

        results[(z, 0)] = ratio_mask_0.sum().cpu().item() / pixels
        results[(z, 20)] = ratio_mask_20.sum().cpu().item() / pixels
        results[(z, 40)] = ratio_mask_40.sum().cpu().item() / pixels
    return results

corrs = []

for i in tqdm(range(len(ds))):
    ratios = _count_precip_ratios(i)
    corrs.append(ratios)

# %%
np.savez_compressed("altitude_wise_sample_wise_ratios.npz",
                    corrs=corrs)