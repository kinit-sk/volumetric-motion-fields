"""PyTorch Dataset for working with radar files that are in h5 format."""
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, generate_binary_structure
import h5py
import pandas as pd
import os
import psutil
from tqdm import tqdm
from functools import reduce
import time
import ctypes
import multiprocessing as mp



class SlovakDatasetDualPol(Dataset):
    """A dataset for working with Slovak radar files that are in h5 format."""

    ALL_PRODUCTS = ['dBZ', 'dBZv', 'KDP', 'PhiDP', 'RhoHV', 'W', 'ZDR']

    def __init__(
        self,
        split="train",
        path=None,
        metadata_folder=None,
        input_block_length=4,
        prediction_block_length=20,
        transform_to_mmh=True,
        normalization_method='none',
        erosion_mask=True,
        denoise=False,
        products=ALL_PRODUCTS,
        aggregate_vertical=False,
        vertical_chunk_size=1,
        memmaps=False,
        timing=False,
        shared_caching=False,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        split : {'train', 'test', 'valid'}
            The type of the dataset: training, testing or validation.
        path : str
            Path to the data folder.
        input_block_length : int
            The number of frames to be used as input to the models.
        prediction_block_length : int
            The number of frames that are predicted and tested against the
            observations.
        """
        assert path is not None, "No path to radar files provided!"
        assert metadata_folder is not None, "No metadata folder for radar files provided!"

        # Inherit from parent class
        super().__init__()

        self.num_frames_input = input_block_length
        self.num_frames_output = prediction_block_length
        self.num_frames = input_block_length + prediction_block_length

        # Load metadata
        self.metadata = pd.read_csv(os.path.join(metadata_folder, "metadata.csv"), index_col=0)


        self.mask_kb = np.nanmax(np.array(h5py.File(os.path.join(metadata_folder, "mask_kb.h5"))['mask']['data']), axis=0)
        self.mask_kh = np.nanmax(np.array(h5py.File(os.path.join(metadata_folder, "mask_kh.h5"))['mask']['data']), axis=0)
        self.mask_mj = np.nanmax(np.array(h5py.File(os.path.join(metadata_folder, "mask_mj.h5"))['mask']['data']), axis=0)
        self.mask_sl = np.nanmax(np.array(h5py.File(os.path.join(metadata_folder, "mask_sl.h5"))['mask']['data']), axis=0)

        self.split = split

        self.metadata.datetime = pd.to_datetime(self.metadata.datetime, format="%Y%m%d%H%M")

        if split == 'test':
            self.datelist = pd.to_datetime(self.metadata[(self.metadata.seq_start == True) & (self.metadata.datetime >= '2022-01-01')].datetime)
            self.seq_start_indices = np.array(self.metadata[(self.metadata.seq_start == True) & (self.metadata.datetime >= '2022-01-01')].index)
        elif split == 'train':
            self.datelist = pd.to_datetime(self.metadata[(self.metadata.seq_start == True) & (self.metadata.datetime < '2022-01-01') & (self.metadata.group % 20 != 0)].datetime)
            self.seq_start_indices = np.array(self.metadata[(self.metadata.seq_start == True) & (self.metadata.datetime < '2022-01-01') & (self.metadata.group % 20 != 0)].index)
        elif split == 'valid':
            self.datelist = pd.to_datetime(self.metadata[(self.metadata.seq_start == True) & (self.metadata.datetime < '2022-01-01') & (self.metadata.group % 20 == 0)].datetime)
            self.seq_start_indices = np.array(self.metadata[(self.metadata.seq_start == True) & (self.metadata.datetime < '2022-01-01') & (self.metadata.group % 20 == 0)].index)
        elif split == 'full':
            self.datelist = pd.to_datetime(self.metadata[self.metadata.seq_start == True].datetime)
            self.seq_start_indices = np.array(self.metadata[self.metadata.seq_start == True].index)
        else:
            raise ValueError("split \"" + self.split  + "\" not available")
        
        indices = []
        for i in range(self.num_frames):
            indices.append(self.seq_start_indices + i)
        self.full_indices = reduce(np.union1d, indices)
        self.seq_start_indices_remap = np.searchsorted(self.full_indices, self.seq_start_indices)

        self.path = path

        if aggregate_vertical:
            assert vertical_chunk_size >= 1 and vertical_chunk_size <= 16, "vertical_chunk_size must be between 1 and 16"
            assert 16 % vertical_chunk_size == 0, "vertical_chunk_size must be a divisor of 16"

            if vertical_chunk_size > 1:
                self.image_size = (16 // vertical_chunk_size, 517, 755)
            else:
                self.image_size = (517, 755)
        else:
            self.image_size = (16, 517, 755)

        self.transform_to_mmh = transform_to_mmh
        self.erosion_mask = erosion_mask
        self.denoise = denoise

        self.common_time_index = self.num_frames_input - 1

        if normalization_method not in ["log", "dBR", "none"]:
            raise NotImplementedError(
                f"data normalization method {normalization_method} not implemented"
            )
        else:
            self.normalization = normalization_method
        
        self.timestep = 5
        self.memmaps = memmaps
        if self.memmaps and not denoise:
            raise ValueError("Denoise must be enabled when using memmaps.")
        if self.memmaps and products != ["dBZ"]:
            raise ValueError("Memmaps are only available for dBZ product.")

        self.products = products
        self.aggregate_vertical = aggregate_vertical
        self.vertical_chunk_size = vertical_chunk_size
        self.timing = timing

        self.shared_caching = shared_caching

        if self.shared_caching:
            full_dataset_base = mp.Array(ctypes.c_float, len(self.full_indices) * reduce(lambda x, y: x * y, self.image_size))
            full_dataset = np.ctypeslib.as_array(full_dataset_base.get_obj())
            full_dataset = full_dataset.reshape(len(self.full_indices), *self.image_size)
            self.full_dataset = torch.from_numpy(full_dataset)

            cached_base = mp.Array(ctypes.c_bool, len(self.full_indices))
            cached = np.ctypeslib.as_array(cached_base.get_obj())
            self.cached = torch.from_numpy(cached)

        


    def __len__(self):
        """Mandatory property for Dataset."""
        return len(self.datelist)

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        t_start = time.time()

        if torch.is_tensor(idx):
            idx = idx.tolist()


        data = torch.empty((self.num_frames, len(self.products),*self.image_size))
        for i, name in enumerate(self.metadata.filename.loc[self.seq_start_indices[idx]:self.seq_start_indices[idx]+self.num_frames-1]):
            fn = os.path.join(self.path, name)

            if self.shared_caching and self.cached[self.seq_start_indices_remap[idx] + i]:
                im = self.full_dataset[self.seq_start_indices_remap[idx] + i]
            else:
                if self.memmaps:
                    fn = fn.replace(".h5", ".npy")
                    im = self.read_memmap(fn)
                else:
                    im = self.read_h5_composite(fn)
                if self.shared_caching:
                    self.full_dataset[self.seq_start_indices_remap[idx] + i] = torch.from_numpy(im)
                    self.cached[self.seq_start_indices_remap[idx] + i] = True
            data[i, ...] = im
            del im
        
        t_loaded = time.time()
        
        # Mask out noise in dBZ
        if self.denoise and ("dBZ" in self.products or "dBZv" in self.products) and not self.memmaps:
            if "RhoHV" in self.products:
                if "dBZ" in self.products:
                    data[:,self.products.index("dBZ")][data[:,self.products.index("RhoHV")] < 0.6] = 0
                if "dBZv" in self.products:
                    data[:,self.products.index("dBZv")][data[:,self.products.index("RhoHV")] < 0.6] = 0
            else:
                print("Warning: RhoHV not in products, not using it to denoise dBZ and/or dBZv.")
            if "dBZ" in self.products:
                for i in range(len(data)):
                    data[i,self.products.index("dBZ")] = self.opening_denoise(data[i,self.products.index("dBZ")])
            if "dBZv" in self.products:
                for i in range(len(data)):
                    data[i,self.products.index("dBZv")] = self.opening_denoise(data[i,self.products.index("dBZv")])
        
        t_denoised = time.time()

        data = self.mask_outside_range(data, idx)

        t_masked = time.time()
        
        inputs, outputs = self.postprocessing(data)

        t_postprocessed = time.time()

        if self.timing:
            print(f"Loading time: {t_loaded - t_start:.2f}s, Denoising time: {t_denoised - t_loaded:.2f}s, Masking time: {t_masked - t_denoised:.2f}s, Postprocessing time: {t_postprocessed - t_masked:.2f}s")
            print(f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f}")

        del data
        return inputs, outputs, idx
    
    def dbz_to_mmh(self, data):
        data = 10 ** (data * 0.1)
        data =  (data / 200) ** (1 / 1.6)
        return data
    
    def mmh_to_dbz(self, data):
        data[data < 0] = 0  # Avoid negative mmh values
        data = 200 * data ** (1.6)
        data = 10 * torch.log10(data + 1)
        return data

    def from_transformed(self, data, scaled=True):
        if scaled:
            if "dBZ" in self.products:
                data[:,self.products.index("dBZ")] = self.invScaler(data[:,self.products.index("dBZ")])
            if "dBZv" in self.products: 
                data[:,self.products.index("dBZv")] = self.invScaler(data[:,self.products.index("dBZv")])
        
        if self.transform_to_mmh:
            if "dBZ" in self.products:
                data[:,self.products.index("dBZ")] = self.mmh_to_dbz(data[:,self.products.index("dBZ")])
            if "dBZv" in self.products:
                data[:,self.products.index("dBZv")] = self.mmh_to_dbz(data[:,self.products.index("dBZv")])

        return data
    
    def postprocessing(self, data: torch.Tensor):
        if self.transform_to_mmh:
            if "dBZ" in self.products:
                data[:,self.products.index("dBZ")] = self.dbz_to_mmh(data[:,self.products.index("dBZ")])
            if "dBZv" in self.products:
                data[:,self.products.index("dBZv")] = self.dbz_to_mmh(data[:,self.products.index("dBZv")])

        # to log-transformed
        data = self.scaler(data)

        # Divide to input & output
        if self.num_frames_output == 0:
            inputs = data
            outputs = torch.empty((0, *data.shape))
        else:
            inputs = data[: -self.num_frames_output, ...]
            outputs = data[-self.num_frames_output :, ...]

        return inputs, outputs

    def mask_outside_range(self, data, idx):
        kb = self.metadata.loc[self.full_indices[self.seq_start_indices_remap[idx]]:self.full_indices[self.seq_start_indices_remap[idx]]+self.num_frames].kb
        kh = self.metadata.loc[self.full_indices[self.seq_start_indices_remap[idx]]:self.full_indices[self.seq_start_indices_remap[idx]]+self.num_frames].kh
        mj = self.metadata.loc[self.full_indices[self.seq_start_indices_remap[idx]]:self.full_indices[self.seq_start_indices_remap[idx]]+self.num_frames].mj
        sl = self.metadata.loc[self.full_indices[self.seq_start_indices_remap[idx]]:self.full_indices[self.seq_start_indices_remap[idx]]+self.num_frames].sl

        maxmask = np.ones(data[0].shape, dtype='uint8')

        diamond = generate_binary_structure(rank=2, connectivity=1)
        if len(data[0].shape) == 3: # for 3D data also perform only 2D erosion equivalent
            diamond = np.expand_dims(diamond, axis=0)
        if len(data[0].shape) == 4: # for 4D data also perform only 2D erosion equivalent
            diamond = np.expand_dims(diamond, axis=(0,1))

        for i in range(data.shape[0]):
            mask = np.zeros(data[i].shape, dtype='uint8')
            if kb.iloc[i]:
                mask = mask | self.mask_kb
            if kh.iloc[i]:
                mask = mask | self.mask_kh
            if mj.iloc[i]:
                mask = mask | self.mask_mj
            if sl.iloc[i]:
                mask = mask | self.mask_sl
            
            maxmask = maxmask & mask

            if i >= self.num_frames_input and self.erosion_mask:
                maxmask = binary_erosion(maxmask, iterations=5, structure=diamond)

            data[i][maxmask == 0] = np.nan
        
        del maxmask, mask
        return data
    

    def opening_denoise(self, data):

        diamond = generate_binary_structure(rank=2, connectivity=1)
        if len(data.shape) == 3: # for 3D data also perform only 2D opening and dilation
            diamond = np.expand_dims(diamond, axis=0)

        high_mask = data > 40
        high_mask = torch.Tensor(binary_dilation(high_mask, structure=diamond, iterations=3))

        opening_mask = data != 0
        
        opening_mask = torch.Tensor(binary_opening(opening_mask, iterations=4, structure=diamond))
        data[(opening_mask == False) & (high_mask == False) & (np.isnan(data) == False)] = 0

        return data


    def get_common_time(self, index):
        return self.datelist.iloc[index]
    
    def scaler(self, data: torch.Tensor):
        if self.normalization == "log":
            return torch.log(data + 0.01)
        if self.normalization == "dBR":
            zeros = data < 0.1
            data[~zeros] = 10.0 * torch.log10(data[~zeros])
            data[zeros] = -15.0
            return data
        if self.normalization == "none":
            return data

    def invScaler(self, data: torch.Tensor):
        if self.normalization == "log":
            return torch.exp(data) - 0.01
        if self.normalization == "dBR":
            data = 10.0 ** (data / 10.0)
            data[data < 0.1] = 0
            return data
        if self.normalization == "none":
            return data


    def read_h5_composite(self, filename):
        """"Read h5 composite."""
        data_out = np.empty((len(self.products), *self.image_size))
        with h5py.File(filename, "r") as hf:
            hf = h5py.File(filename, "r")
            for i, product in enumerate(self.products):
                data = np.array(hf[product]['data']['data'], dtype=float)
                datamax = hf[product]['data']['what'].attrs.get('datamax')
                datamin = hf[product]['data']['what'].attrs.get('datamin')
                depth = hf[product]['data']['what'].attrs.get('datadepth')
                data[data == 0] = np.nan
                data = data - 1
                data /= (2 ** depth - 2)
                data *= (datamax - datamin)
                data += datamin
                data[np.isnan(data)] = 0

                # 'dBZ', 'dBZv', 'KDP', 'PhiDP', 'RhoHV', 'W', 'ZDR'
                if self.aggregate_vertical:
                    if (product == self.ALL_PRODUCTS[0] or #dBZ
                        product == self.ALL_PRODUCTS[1] or #dBZv
                        product == self.ALL_PRODUCTS[2]):  #KDP
                        data = np.nanmax(data, axis=0)
                    elif product == self.ALL_PRODUCTS[3]:  #PhiDP
                        data = np.nanmean(data, axis=0)
                    elif product == self.ALL_PRODUCTS[4]:  #RhoHV
                        data = np.nanmax(data, axis=0)
                    elif product == self.ALL_PRODUCTS[5]:  #W
                        data = np.nanmax(data, axis=0)
                    elif product == self.ALL_PRODUCTS[6]:  #ZDR
                        data = np.nanmedian(data, axis=0)
                
                data_out[i] = data

        return torch.tensor(data_out)
    
    def read_memmap(self, filename):
        """"Read h5 composite."""
        try:
            data = np.memmap(filename, dtype='uint8', mode='r', shape=(16, 517, 755))
        except Exception as e:
            print(f"Error loading memmap {filename}. Retrying...")
            time.sleep(1)
            data = np.memmap(filename, dtype='uint8', mode='r', shape=(16, 517, 755))
        
        product = 'dBZ'  # Assuming we are reading dBZ product from memmap
        datamax = 95.5
        datamin = -31.5
        depth = 8

        data = np.array(data, dtype=float)
        data[data == 0] = np.nan
        data = data - 1
        data /= (2 ** depth - 2)
        data *= (datamax - datamin)
        data += datamin
        data[np.isnan(data)] = 0

        if self.aggregate_vertical:
            if self.vertical_chunk_size > 1:
                new_shape = [data.shape[0] // self.vertical_chunk_size, self.vertical_chunk_size] + list(data.shape[1:])
                data = data.reshape(new_shape)
                data = np.nanmax(data, axis=1)
            else:
                data = np.nanmax(data, axis=0)
                
        return torch.tensor(data).unsqueeze(dim=0)
