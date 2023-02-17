# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 09:35:25 2021

@author: chong
"""

from swmm_api import read_out_file
import pandas as pd

out=read_out_file('./sim/staf.out')
print(out.labels)
#print(out.columns_raw)