import random
from pathlib import Path
from typing import List, Optional

import torch
import pytorch_lightning as pl
import lightning.pytorch as plv2

from sen12tp.dataset import Patchsize, SEN12TP, FilteredSEN12TP
from sen12tp.constants import (
    MIN_VV_VALUE,
    MIN_VH_VALUE,
    MIN_DEM_VALUE,
    MAX_DEM_VALUE,
    cgls_simplified_mapping,
)
from sen12tp.constants import BandNames
from sen12tp.utils import default_clipping_transform

'''
# this was the original version before i modified it to 
# create the validation set
def create_sen12tp_datasets(self):
    sen12tp_kwargs = {
        "patch_size": self.patch_size,
        "transform": self.transform,
        "model_targets": self.model_targets,
        "clip_transform": self.clipping_method,
        "model_inputs": self.model_inputs,
        "end_transform": self.end_transform,
        "stride": self.stride,
    }
    sen12tp_train_ds = SEN12TP(self.dataset_dir / "train", **sen12tp_kwargs)
    #sen12tp_val_ds = SEN12TP(self.dataset_dir / "val", **sen12tp_kwargs)
    sen12tp_test_ds = SEN12TP(self.dataset_dir / "test", **sen12tp_kwargs)
    random.shuffle(sen12tp_train_ds.patches)
    #random.shuffle(sen12tp_val_ds.patches)
    self.sen12tp_train = FilteredSEN12TP(sen12tp_train_ds, shuffle=self.shuffle_train)
    #self.sen12tp_val = FilteredSEN12TP(sen12tp_val_ds)
    self.sen12tp_test = FilteredSEN12TP(sen12tp_test_ds)
'''

def split_patches(patches, split_ratio=0.8):
    split_idx = int(len(patches) * split_ratio)
    return patches[:split_idx], patches[split_idx:]

def create_sen12tp_datasets(self):
    sen12tp_kwargs = {
        "patch_size": self.patch_size,
        "transform": self.transform,
        "model_targets": self.model_targets,
        "clip_transform": self.clipping_method,
        "model_inputs": self.model_inputs,
        "end_transform": self.end_transform,
        "stride": self.stride,
    }
    # this was created using chatGPT to modify the above function to
    # also create a validation set out of the training set
    
    # Create dataset objects
    sen12tp_train_ds = SEN12TP(self.dataset_dir / "train", **sen12tp_kwargs)
    sen12tp_test_ds = SEN12TP(self.dataset_dir / "test", **sen12tp_kwargs)

    # Split patches into train and validation sets
    random.shuffle(sen12tp_train_ds.patches)
    train_patches, val_patches = split_patches(sen12tp_train_ds.patches)  # Define this function

    # Create separate dataset instances using filtered patches
    sen12tp_train_ds.patches = train_patches
    sen12tp_val_ds = SEN12TP(self.dataset_dir / "train", **sen12tp_kwargs)  # Create a new instance
    sen12tp_val_ds.patches = val_patches  # Assign the validation patches

    # Assign datasets
    self.sen12tp_train = FilteredSEN12TP(sen12tp_train_ds, shuffle=self.shuffle_train)
    self.sen12tp_val = FilteredSEN12TP(sen12tp_val_ds)
    self.sen12tp_test = FilteredSEN12TP(sen12tp_test_ds)



def create_dataloader(self, dataset, drop_last=False):
    assert dataset, ("Run setup() before calling dataloader()!")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        drop_last=drop_last,
    )


class SEN12TPDataModuleV2(plv2.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = "path/to/dir",
        batch_size: int = 32,
        patch_size: Patchsize = Patchsize(256, 256),
        stride: int = 249,
        model_inputs: List[str] = None,
        model_targets: List[str] = None,
        transform=None,
        end_transform=None,
        num_workers: int = 1,
        pin_memory: bool = True,
        shuffle_train: bool = False,
        drop_last_train: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.clipping_method = default_clipping_transform
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.model_inputs = (
            model_inputs if model_inputs else ["VV_corrected", "VH_corrected"]
        )
        self.model_targets = model_targets if model_targets else ["NDVI"]
        self.sen12tp_train: FilteredSEN12TP
        #self.sen12tp_val: FilteredSEN12TP
        self.sen12tp_test: FilteredSEN12TP
        self.num_workers = num_workers
        self.end_transform = end_transform
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train

    def setup(self, stage: Optional[str] = None):
        create_sen12tp_datasets(self)

    def train_dataloader(self):
        return create_dataloader(self, self.sen12tp_train, drop_last=self.drop_last_train)

    def val_dataloader(self):
        return create_dataloader(self, self.sen12tp_val)

    def test_dataloader(self):
        return create_dataloader(self, self.sen12tp_test)


class SEN12TPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = "path/to/dir",
        batch_size: int = 32,
        patch_size: Patchsize = Patchsize(256, 256),
        stride: int = 249,
        model_inputs: List[str] = None,
        model_targets: List[str] = None,
        transform=None,
        end_transform=None,
        num_workers: int = 1,
        pin_memory: bool = True,
        shuffle_train: bool = False,
        drop_last_train: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.clipping_method = default_clipping_transform
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.model_inputs = (
            model_inputs if model_inputs else ["VV_corrected", "VH_corrected"]
        )
        self.model_targets = model_targets if model_targets else ["NDVI"]
        self.sen12tp_train: FilteredSEN12TP
        #self.sen12tp_val: FilteredSEN12TP
        self.sen12tp_test: FilteredSEN12TP
        self.num_workers = num_workers
        self.end_transform = end_transform
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train

    def setup(self, stage: Optional[str] = None):
        create_sen12tp_datasets(self)

    def train_dataloader(self):
        return create_dataloader(self, self.sen12tp_train, drop_last=self.drop_last_train)

    def val_dataloader(self):
        return create_dataloader(self, self.sen12tp_val)

    def test_dataloader(self):
        return create_dataloader(self, self.sen12tp_test)
