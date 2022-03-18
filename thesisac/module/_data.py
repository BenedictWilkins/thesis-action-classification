#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pytorch_lightning as pl
import pathlib
import gymu
import yaml
import glob
import numpy as np

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import List, Dict, Any,Union

import logging
Logger = logging.getLogger(__name__.split(".")[0])

__all__ = ("ACDataModule", "download_from_kaggle", "configure_environment")

ROOT_PATH = pathlib.Path("~/.data/").expanduser().resolve()
ROOT_PATH.mkdir(parents=True, exist_ok=True)

class ACDataModule(pl.LightningDataModule):

   def __init__(self, 
                  path : Union[pathlib.Path, str] = None,
                  env_id : str = None,
                  env_kwargs : str = {}, 
                  policy : str = 'gymu.policy.Uniform',
                  policy_kwargs : Dict[str,Any] = {}, 
                  data_split : List[float] = [0.7,0.2,0.1],
                  batch_size : int = 256, 
                  train_mode : Union[str, List[str]] = ['state', 'action'],
                  validate_mode : Union[str, List[str]] = ['state', 'action'],
                  test_mode : Union[str, List[str]] = ['state', 'action'],
                  num_episodes : int = -1, 
                  shuffle_buffer_size : int = 10000,
                  initial_buffer_size : int = 10000,
                  num_workers : int = 12,
                  prefetch_factor : int = 2,
                  window_size : int = 1,
                  in_memory : int = False,
                  **kwargs):
                  
      self.data_split = np.array(data_split)
      self.batch_size = batch_size
      self.train_data, self.validate_data, self.test_data = None, None, None
      self.train_mode, self.validate_mode, self.test_mode = self.resolve_modes(train_mode, validate_mode, test_mode)
      self.num_episodes = num_episodes
      self.shuffle_buffer_size = shuffle_buffer_size
      self.initial_buffer_size = initial_buffer_size
      self.num_workers = num_workers
      self.prefetch_factor = prefetch_factor
      self.window_size = window_size
      self.in_memory = in_memory
      self.persistent_workers = self.num_workers > 0

      if path is not None:
         self.path = pathlib.Path(path)
         self.train_files, self.validate_files, self.test_files = self.resolve_files(self.path) 
      elif env_id is not None:
         self.env_id = env_id
         self.env_kwargs = env_kwargs
         self.policy = policy
         self.policy_kwargs = policy_kwargs
      elif env_id is None and path is None:
         raise ValueError("One of 'path' or 'env_id' must specified.")
      
      assert self.window_size >= 1

   @property
   def is_live(self): # is this dataset using an environment rather data on disk...
      return not hasattr(self, "path")

   def resolve_modes(self, *modes):
      return [(gymu.mode.mode(m) if m is not None else None) for m in modes]

   def resolve_files(self, path):
      """ 
         Search for train, validate and test directories, otherwise split the data according to cfg.dataset.split.
         Args:
            path (pathlib.Path, str): path to search.
      """
      path = pathlib.Path(path)
      if not path.exists():
         raise ValueError(f"Dataset path: '{path}' doesnt exist.")
      def _find(dirs, *labels):
         found = next((v for k,v in dirs.items() if k in labels), None)
         if found is not None:
            found = pathlib.PurePath(path, "**/*.tar")
            return list(sorted(glob.glob(str(found), recursive=True)))
         return []
      dirs = {f.name:f for f in path.iterdir() if f.is_dir()}
      train_files = _find(dirs, 'train', 'training')
      test_files = _find(dirs, 'test', 'testing')
      validate_files = _find(dirs, 'val', 'validate', 'validation')
      if len(train_files + test_files + validate_files) == 0: 
         # no directories were found... use files from the current folder and manually splitting.
         path = pathlib.PurePath(path, "**/*.tar*")
         Logger.info(f"Searching for data files: {path}")
         files = list(sorted(glob.glob(str(path), recursive=True)))
         assert len(files) > 0 # didn't find any files...
         split = np.cumsum(self.data_split * len(files)).astype(np.int64)[:-1]
         train_files, validate_files, test_files = [x.tolist() for x in np.split(np.array(files), split)]

      assert len(train_files + validate_files + test_files) > 0
      assert len(train_files) > 0
      Logger.info(f"Found: {len(files)} files, {len(train_files)} train files, {len(validate_files)} validate files, {len(test_files)} test files.")
      return train_files, validate_files, test_files

   def prepare_file_data(self, files):
      if len(files) == 0:
         return None # dont use this kind of data...? 
      dataset = gymu.data.dataset(files, shardshuffle=True)
      dataset = dataset.gymu.decode(keep_meta=False)
      return dataset

   def prepate_env_data(self):
      raise NotImplementedError("TODO")
      
   def prepare_base_data(self, dataset, mode, in_memory, shuffle):
      if dataset is None or mode is None:
         return None
      if self.window_size > 1:
         # keep only mode keys + done, this is important for not overlapping episodes and efficiency when stacking
         dataset = dataset.gymu.keep(set([*mode.keys(), *(['done'] if self.window_size > 1 else [])]))
         dataset = dataset.gymu.window(self.window_size)
      dataset = dataset.gymu.mode(mode)
      dataset = dataset.shuffle(self.shuffle_buffer_size, initial=self.initial_buffer_size) if shuffle and not in_memory else dataset
      dataset = dataset.gymu.to_tensor_dataset(num_workers=self.num_workers, show_progress=True) if in_memory else dataset.batched(self.batch_size)
      #print(dataset.tensors[0].shape)
      return dataset

   def prepare_train_data(self):
      dataset = self.prepare_file_data(self.train_files)
      return self.prepare_base_data(dataset, self.train_mode, self.in_memory, True)

   def prepare_validate_data(self):
      dataset = self.prepare_file_data(self.validate_files)
      return self.prepare_base_data(dataset, self.validate_mode, False, False)
   
   def prepare_test_data(self):
      dataset = self.prepare_file_data(self.test_files)
      return self.prepare_base_data(dataset, self.test_mode, False, False) # doesnt need to be in memory

   def prepare_data(self):
      self.train_data = self.prepare_train_data()
      self.validate_data = self.prepare_validate_data()
      self.test_data = self.prepare_test_data() 

   def train_dataloader(self):
      if self.in_memory:
         return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers, drop_last=True, pin_memory=True)
      else:
         return DataLoader(self.train_data, shuffle=False, batch_size=None, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers, pin_memory=True)

   def val_dataloader(self):
      return DataLoader(self.validate_data, batch_size=None, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

   def test_dataloader(self):
      return DataLoader(self.test_data, batch_size=None, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

         
def download_from_kaggle(path : Union[str, pathlib.Path], urls : Union[List[str],str], force : bool = False):
   from kaggle.api.kaggle_api_extended import KaggleApi
   path = pathlib.Path(path)
   if isinstance(urls, str):
      urls = [urls]
   if len(urls) == 0: raise ValueError("Atleast one download URL must be specified.")
   api = KaggleApi()
   api.authenticate() # requires ~/.kaggle/kaggle.json file
   path.mkdir(parents=True, exist_ok=False) # it shouldnt exist yet...
   for url in urls:
      api.dataset_download_files(url, quiet=False, unzip=True, force=force, path=str(path))

def configure_environment(cfg):
   # cfg should be the dataset configuration and contain either path or env_id.
   def _strip_yaml_tags(yaml_data):
      result = []
      for line in yaml_data.splitlines():
         idx = line.find("!")
         if idx > -1:
            line = line[:idx]
         result.append(line)
      return '\n'.join(result)

   if not 'path' in cfg:
      kwargs = cfg.get("env_kwargs", OmegaConf.create())
      kwargs = OmegaConf.to_container(kwargs)
      env = gymu.make(cfg.env_id, **kwargs)
      from thesisdata.dataset import get_environment_config
      yaml_data = yaml.dump(get_environment_config(env))
      if hasattr(env, "close"):
         env.close()
      return OmegaConf.create(_strip_yaml_tags(yaml_data))
 
   def _dump_yaml(path):
      path_meta = pathlib.Path(path, "meta.yaml")
      with path_meta.open('r') as f:
         return OmegaConf.create(_strip_yaml_tags(yaml.dump(yaml.safe_load(f))))

   path = pathlib.Path(ROOT_PATH, cfg.path).expanduser()
   Logger.info(f"Configuring dataset for path: {path}")

   if not path.exists() or next(path.iterdir(), None) is None: # path is empty, either construct the dataset, use live or download from kaggle
      if 'kaggle' in cfg:
         download_from_kaggle(ROOT_PATH, cfg.kaggle.urls, force = cfg.kaggle.get('force', False))
         if not path.exists() or next(path.iterdir(), None) is None:
            raise ValueError(f"Downloaded data path {ROOT_PATH} does not match path specified {path}.")
   return _dump_yaml(path)


