#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn

import tml
from tml import ResBlock2D, View
from tml.utils import as_shape

import torchvision

import pytorch_lightning as pl
from hydra.utils import instantiate

from pprint import pprint

# type hints
from typing import Union, List, Tuple, Dict
from torch import Tensor


class ACLightningModule(pl.LightningModule):
    
    def __init__(self, model, criterion, optimiser, metrics=[]):
        super().__init__()
        self.model = instantiate(model)
        self.criterion = instantiate(criterion)
        self.metrics = [instantiate(m) for m in metrics]
        self.optimiser =  instantiate(optimiser, self.parameters())

    def forward(self, state, action):
        
  
        return self.model(state, action[:,1:-1]) # TODO supply correct arguments to model? 

    def training_step(self, batch : Tuple[Tensor,Tensor], _):
        """ The goal is to predict the first action (A_0) in a sequence {S_0, A_0, S_1, A_1, ... S_N, A_N}.

        Args:
            batch (Tuple[Tensor,Tensor]): state, action tensors
            batch_index (int): ignore.
        """
        state, action = batch
        p_action = self.forward(state, action)
        g_action = torch.nn.functional.one_hot(action[:,0], p_action.shape[-1]).float()
        
        loss = self.criterion(p_action, g_action)
        self.log("train/loss", loss.item())
        return loss

    def validation_step(self, batch, _):
        state, action = batch
        p_action = self.forward(state, action)
        g_action = torch.nn.functional.one_hot(action[:,0], p_action.shape[-1]).float()
        if "Logit" in self.criterion.__class__.__name__:  # TODO crude hack...
            p_action = torch.softmax(p_action, dim=-1)
        for metric in self.metrics: 
            self.log(f"validation/{metric.__class__.__name__}", metric(g_action, p_action))

    def configure_optimizers(self):
        return self.optimiser