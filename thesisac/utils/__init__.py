#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import hydra
import omegaconf
import pathlib
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

def load_model(run, cfg):
    model_class = hydra.utils.get_class(cfg.lightning._target_)
    model_artifacts = [art for art in run.logged_artifacts() if art.type == "model"]
    assert len(model_artifacts) > 0 # no models were found?
    model_artifact = next(model for model in model_artifacts if cfg.wandb.model_alias in model.aliases)
    model_path = model_artifact.download()
    model_path = str(pathlib.Path(model_path, "model.ckpt").resolve())
    print(f"Found model at: {model_path}")
    return model_class.load_from_checkpoint(model_path, **cfg.lightning)

@hydra.main(config_name="config_test.yaml", config_path="./configuration")
def main(cfg) -> None:
    api = wandb.Api()
    #print(list(api.runs("benedict-wilkins/WOB-Experiments")))
    run = api.run(str(pathlib.PurePath(cfg.wandb.run)))
    run.config['wandb'] = {**cfg.wandb}
    run.config['data']['path'] = "${hydra:runtime.cwd}/dataset"
    run.config['data']['num_workers'] = 12 # CHANGE FOR LOCAL NEEDS
    #cfg = OmegaConf.update(cfg, run.config)

    cfg = omegaconf.OmegaConf.create(run.config)
    cfg.data.train_files = "NORMAL-TRAIN/*/---.tar"
    cfg.data.validation_files = "NORMAL-TRAIN-SMALL/*/----.tar"
    cfg.data.test_files = "TEST/*/ep-0000/*.tar"

    
    #cfg.data.test_files = "TEST/ScreenTear/ep-0000/*.tar" 
    #print(cfg)

    # add the option of changing the score when testing...

    model = load_model(run, omegaconf.OmegaConf.create(run.config))
    
    cfg.data.train_files = None # dont load any training data...
    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()

if __name__ == "__main__":
    main()
