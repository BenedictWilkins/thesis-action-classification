#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import logging
import sys

__all__ = ("Log", )
Log = logging.getLogger(__name__.split(".")[0])
Log.setLevel(logging.DEBUG)
#streamHandler = logging.StreamHandler(sys.stdout)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#streamHandler.setFormatter(formatter)
#Log.addHandler(streamHandler)