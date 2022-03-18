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

from functools import reduce
from tml.module.MLP import MLP
from tml.module.join import JoinAttention, JoinConcatenate, JoinSum, JoinProd
from tml.utils.shape import as_shape

__all__ = ("MLPClassifier",)

DEFAULT_JOIN = lambda x: torch.cat(x, dim=1)

class MLPClassifier(nn.Module):

   def __init__(self, input_shape, hidden_shape, output_shape, join=DEFAULT_JOIN, layers=2, output_activation=None):
      super().__init__()
      input_shape = as_shape(input_shape)
      assert len(input_shape) > 1 # input shape requires atleast 2 dimensions (window_size, *input_shape)
      (self.window_size, *self.input_shape) = input_shape
      self.hidden_shape = as_shape(hidden_shape)
      self.output_shape = as_shape(output_shape)
      self.encoder = MLP(self.input_shape, self.hidden_shape, self.hidden_shape, layers=layers, output_activation=nn.LeakyReLU)
      self.join = join
      with torch.no_grad():
         join_shape = as_shape(self.join([torch.empty(2, *self.hidden_shape)]*self.window_size).shape[1:])
      #print(join_shape, self.hidden_shape, self.output_shape)
      self.classifier = MLP(join_shape, self.hidden_shape, self.output_shape, layers=2)
      self.output_activation = output_activation

   def forward(self, x, *args): # expects [batch_size, window_size, *input_shape]
      x = x.transpose(0,1) # [window_size, batch_size, *input_shape]
      ys = self.join([self.encoder(z) for z in x]) # [batch_size, *join_shape]
      ys = self.classifier(ys) # [batch_size, *output_shape]
      return ys if self.output_activation is None else self.output_activation(ys) 