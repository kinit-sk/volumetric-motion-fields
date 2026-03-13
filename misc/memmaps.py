from tqdm import tqdm
import os
os.chdir("/home/ppavlik/repos/nowcasting-SVK-dualpol")

import numpy as np

import datasets.SlovakDatasetDualPol as SlovakDatasetDualPol

ds = SlovakDatasetDualPol(path=fr"/projects/p709-24-2/datasets/SHMU_4_New",
                   metadata_folder=fr"/projects/p709-24-2/datasets/SHMU_4_New/metadata",
                   split="test", denoise=True, erosion_mask=False, transform_to_mmh=False,
                   input_block_length=8, prediction_block_length=16,
                   products=['dBZ', 'RhoHV'], aggregate_vertical=False, timing=True)

def convert_float_to_uint(data, datamin=-31.5, datamax=95.5, depth=8, target_type=np.uint8):
    """
    Convert float array to uint array.
    
    """
    datamin = float(datamin)
    datamax = float(datamax)
    
    maskmin = np.nonzero(data < datamin)
    maskmax = np.nonzero(data > datamax)
    
    data[maskmin] = datamin
    data[maskmax] = datamax
    
    new_data = np.round(((data - datamin) / (datamax - datamin) * (2 ** depth - 2)) + 1)
    new_data[np.isnan(new_data)] = 0
    new_data = new_data.astype(target_type)
    
    return new_data

for name in tqdm(ds.metadata.filename, total=len(ds.metadata.filename)):
    fn = os.path.join(ds.path, name)
    data = ds.read_h5_composite(fn)
    
    data[ds.products.index("dBZ")][data[ds.products.index("RhoHV")] < 0.6] = 0

    data[ds.products.index("dBZ")] = ds.opening_denoise(data[ds.products.index("dBZ")])

    data = data[0]  # Select only dBZ

    data = convert_float_to_uint(data)

    fp = np.memmap("/projects/p709-24-2/datasets/SHMU-4-3Dmemmap/" + name.split(".")[0] + ".npy", dtype='uint8', mode='w+', shape=data.shape)
    fp[:] = data[:]