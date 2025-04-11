import random
from pathlib import Path
from typing import List, Optional

import torch
import pytorch_lightning as pl
import lightning.pytorch as plv2

from sen12tp_cov.dataset import Patchsize, SEN12TP, FilteredSEN12TP
from sen12tp_cov.constants import (
    MIN_VV_VALUE,
    MIN_VH_VALUE,
    MIN_DEM_VALUE,
    MAX_DEM_VALUE,
    cgls_simplified_mapping,
)
from sen12tp_cov.constants import BandNames
from sen12tp_cov.utils import default_clipping_transform


'''
# this was the original version before i modified it to 
# create the validation set
# this doesn't work becaues there is no longer a separate
# validation set in the 
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

'''
# this was my original re-implementation
# but i'm suspecting that its messing everything up
# because the models don't look like they're
# that well trained

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
'''
# This is an attempt to fix it to do what Thomas says
def create_sen12tp_datasets(self):
    from collections import defaultdict
    import random

    sen12tp_kwargs = {
        "patch_size": self.patch_size,
        "transform": self.transform,
        "model_targets": self.model_targets,
        "clip_transform": self.clipping_method,
        "model_inputs": self.model_inputs,
        "end_transform": self.end_transform,
        "stride": self.stride,
    }

    # Load full dataset
    sen12tp_train_ds = SEN12TP(self.dataset_dir / "train", **sen12tp_kwargs)
    sen12tp_test_ds = SEN12TP(self.dataset_dir / "test", **sen12tp_kwargs)

    all_patches = sen12tp_train_ds.patches
    print(f"📦 Total patches loaded: {len(all_patches)}")

    # Group patches by scene
    scene_to_patches = defaultdict(list)
    for patch in all_patches:
        scene_name = patch[0]['s1'].parent.name  # or whatever uniquely identifies the scene
        scene_to_patches[scene_name].append(patch)

    print(f"🗺️ Total unique scenes: {len(scene_to_patches)}")

    # Split scenes into train and val
    all_scenes = list(scene_to_patches.keys())
    random.shuffle(all_scenes)
    val_scene_count = int(len(all_scenes) * 0.11)

    val_scenes = set(all_scenes[:val_scene_count])
    train_scenes = set(all_scenes[val_scene_count:])

    print(f"📚 Training scenes: {len(train_scenes)}")
    print(f"🧪 Validation scenes: {len(val_scenes)}")

    # Assign patches based on scene split
    train_patches = []
    val_patches = []

    for scene, patches in scene_to_patches.items():
        if scene in val_scenes:
            val_patches.extend(patches)
        else:
            train_patches.extend(patches)

    print(f"✅ Final train patches: {len(train_patches)}")
    print(f"✅ Final val patches: {len(val_patches)}")

    # Double-check: ensure no scene overlap
    assert not (train_scenes & val_scenes), "🚨 Scene overlap between train and val!"
    assert all(p[0]['s1'].parent.name in train_scenes for p in train_patches)
    assert all(p[0]['s1'].parent.name in val_scenes for p in val_patches)

    # Assign patches to dataset instances
    sen12tp_train_ds.patches = train_patches
    sen12tp_val_ds = SEN12TP(self.dataset_dir / "train", **sen12tp_kwargs)
    sen12tp_val_ds.patches = val_patches

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
