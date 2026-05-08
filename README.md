# Assessing the Utility of Volumetric Motion Fields for Radar-based Precipitation Nowcasting with Physics-informed Deep Learning
Repository containing the code needed to replicate the volumetric motion field estimator from the paper [*Assessing the Utility of Volumetric Motion Fields for Radar-based Precipitation Nowcasting with Physics-informed Deep Learning*](https://arxiv.org/abs/2603.13589).

## Requirements
To run this code, you need Python 3.13+ and the following dependencies:

PyTorch, PyTorch Lightning, Matplotlib, Pandas, IPython kernel, Pysteps and wandb

To install, run: ```conda create --name <env> --file requirements.txt```

## Sample Demo

The trained 3DMF-U-Net model checkpoint is included in the repository along with a single data sample for demonstration. To visualize the model's output, run the Jupyter notebook located in `notebooks\3DMF-U-Net_visualize.ipynb`. The notebook generates multiple interactive HTML animations. For more test samples, download the radar data file from the link below and paste the files into the `data\radar_files` folder.

The demo notebook was verified to run on **NVIDIA GeForce RTX 4070 Laptop GPU with 8GB of VRAM**.

## Replication

To train the model yourself, download the dataset from the link below and paste the files into the `data\radar_files` folder. To train the volumetric 3DMF-U-Net model, simply run:

```python train_model.py MF-3D-U-Net_altitude_wise```

If you want to also train the baseline model processing the CMAX radar images, run:

```python train_model.py MF-3D-U-Net```

The training is configured to run on a single **NVIDIA A100 40 GB**.

A test run is automatically performed after the training ends using the checkpoint that performed the best on the validation set.

## Links
### Dataset archive: [https://doi.org/10.5281/zenodo.20077116](https://doi.org/10.5281/zenodo.20077116)
- Data from the Slovak radar network - four dual-pol doppler radars
- roughly 3.5 years
- quantized, zipped and containing only the volumetric reflectivity fields, the size is 13 GB
