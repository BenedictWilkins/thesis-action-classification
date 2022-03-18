#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 18-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


import wandb
import pathlib

from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
Logger = logging.getLogger(__name__.split(".")[0])

from ..module import ACLightningModule, ACDataModule

def _run_decorate(fun):
    def _decorate(run, *args, **kwargs):
        if isinstance(run, str):
            api = wandb.Api()
            run = api.run(str(pathlib.PurePath(run)))
        if kwargs.get('cfg', None) is None:
            kwargs['cfg'] = OmegaConf.create(run.config)
        return fun(run, *args, **kwargs)
    return _decorate

@_run_decorate
def load_lightning_module(run, cfg=None, model_alias='best'):
    model_class = ACLightningModule
    model_artifacts = [art for art in run.logged_artifacts() if art.type == "model"]
    assert len(model_artifacts) > 0 # no models were found?
    model_artifact = next(model for model in model_artifacts if model_alias in model.aliases)
    model_path = model_artifact.download()
    model_path = str(pathlib.Path(model_path, "model.ckpt").resolve())
    Logger.info(f"Found model at: {model_path}")
    return model_class.load_from_checkpoint(model_path, model=cfg.model, optimiser=cfg.optimiser, criterion=cfg.criterion, metrics=cfg.get('metrics', []))

def load_data_module(run, cfg=None):
    return ACDataModule(**OmegaConf.to_container(cfg.dataset))

@_run_decorate
def load_run(run, cfg=None, **kwargs): 
    Logger.info(f"Loading run from: {run}")
    del cfg.trainer.logger
    # TODO resolve cfg.dataset.path if needed (assume the dataset is $HOME/.data)

    module = load_lightning_module(run, cfg=cfg, model_alias=kwargs.get('model_alias', 'best'))
    data_module = load_data_module(run, cfg=cfg)

    trainer = instantiate(cfg.trainer)
    return trainer, module, data_module, cfg