#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import gymu

import hydra
import gym
import pathlib
import yaml
import tml

import wandb

from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from thesisac import configure_environment, ACDataModule, ACLightningModule

# REGISTER RESOLVERS
def omegaconfg_list_merge(x, y):
    if not isinstance(x, ListConfig):
        x = ListConfig([x])
    if not isinstance(y, ListConfig):
        y = ListConfig([y])
    return x + y

OmegaConf.register_new_resolver("merge", omegaconfg_list_merge)
OmegaConf.register_new_resolver("environment", configure_environment)

@hydra.main(config_name="config.yaml", config_path="./configuration")
def main(cfg):
    from pprint import pprint

    # load dataset meta data, download data and update config
    OmegaConf.resolve(cfg)
    pprint(OmegaConf.to_container(cfg))

    data_module = ACDataModule(**OmegaConf.to_container(cfg.dataset))
    module = ACLightningModule(cfg.model, cfg.criterion, cfg.optimiser, cfg.get('metrics', []))

    wandb.init(
        project=cfg.trainer.logger.project, 
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    trainer = instantiate(cfg.trainer)
    trainer.fit(module, datamodule=data_module)
    #trainer.test(model, datamodule=datamodule)

    wandb.finish() # prevents hanging at the end...?

if __name__ == "__main__":
    main()
