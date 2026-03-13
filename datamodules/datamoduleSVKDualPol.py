"""A datamodule for working with Slovak dataset for training, validation and testing that has files in h5 format."""
import torch
import pytorch_lightning as pl
from torchvision.transforms import v2
from torch.utils.data import Sampler, WeightedRandomSampler, DataLoader
from torchvision.transforms.functional import crop, hflip, vflip, rotate

from datasets import SlovakDatasetDualPol
import random

class SetShuffleSampler(Sampler[int]):
    def __init__(self, length):
        self.list = list(range(0,length))
        random.shuffle(self.list)

    def __len__(self) -> int:
        return len(self.list)

    def __iter__(self):
        yield from self.list


class SlovakDataModuleDualPol(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        if stage == "fit":
            self.train_dataset = SlovakDatasetDualPol(
                split="train", **self.dsconfig.SHMUDataset
            )
            self.valid_dataset = SlovakDatasetDualPol(
                split="valid", **self.dsconfig.SHMUDataset
            )
        if stage == "test":
            self.test_dataset = SlovakDatasetDualPol(
                split="test", **self.dsconfig.SHMUDataset
            )
        if stage == "predict":
            self.predict_dataset = SlovakDatasetDualPol(
                split="test", **self.dsconfig.SHMUDataset
            )

    def train_dataloader(self):
        ds = self.train_dataset
        scores = ds.metadata[ds.metadata.index.isin(ds.seq_start_indices)].score

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            pin_memory=False,
            collate_fn=_collate_fn,
            sampler=WeightedRandomSampler(scores ** self.train_params.weights_exp, len(ds)),
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            sampler=SetShuffleSampler(len(self.valid_dataset)),
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            pin_memory=False,
            collate_fn=_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_params.predict_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader
    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y, idx = batch
        if self.trainer.training:
            rand_crop = v2.RandomCrop(size=self.dsconfig.augmentations.crop_size)
            params = rand_crop.get_params(x, output_size=self.dsconfig.augmentations.crop_size)
            x = crop(x, *params)
            y = crop(y, *params)
        else:
            center = v2.CenterCrop(size=self.dsconfig.augmentations.crop_size)
            x = center(x)
            y = center(y)
        return x, y, idx
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y, idx = batch
        if self.trainer.training:
            x, y = self.apply_augments(x, y)
        return x, y, idx
    
    def apply_augments(self, x, y):

        if self.dsconfig.augmentations.horizontal_flip:
            if random.random() >= 0.5:
                x = hflip(x)
                y = hflip(y)

        if self.dsconfig.augmentations.vertical_flip:
            if random.random() >= 0.5:
                x = vflip(x)
                y = vflip(y)
        
        if self.dsconfig.augmentations.rotate:
            angle = random.choice([0, 90, 180, 270])
            x = rotate(x, angle)
            y = rotate(y, angle)
        
        if self.dsconfig.augmentations.edge_nan_padding > 0:
            bordermask = torch.zeros(y.shape[1], y.shape[2], y.shape[3])
            for i in range(0, y.shape[1]):
                border = self.dsconfig.augmentations.edge_nan_padding*(i+1)
                bordermask[i,border:-border,border:-border] = 1
            bordermask = bordermask.unsqueeze(dim=0).repeat((y.shape[0],1,1,1))
            y[bordermask == 0] = float('nan')

        return x, y  

def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

