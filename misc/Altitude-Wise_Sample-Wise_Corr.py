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

def _count_precip_corr(i):
    device = torch.device("cuda:0")
    pearson = PearsonCorrCoef().to(device)

    idx = [i]

    x, _, _ = ds[idx[0]]
    x = x.to(device)

    # number of altitude levels
    n_altitudes = x.shape[2]

    results = {}

    for z1, z2 in combinations(range(n_altitudes), 2):
        p1 = x[0, 0, z1]
        p2 = x[0, 0, z2]

        corr = pearson(
            p1[~torch.isnan(p1)].flatten(),
            p2[~torch.isnan(p2)].flatten()
        )

        results[(z1, z2)] = corr.cpu().item()

    return results

corrs = []

for i in tqdm(range(len(ds))):
    pearson_corrs = _count_precip_corr(i)
    corrs.append(pearson_corrs)

# %%
np.savez_compressed("altitude_wise_sample_wise_eval_corrs_nocrop.npz",
                    corrs=corrs)