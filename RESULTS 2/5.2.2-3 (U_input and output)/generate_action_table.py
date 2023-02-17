# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 22:19:51 2021

@author: chong
"""

#this code is only used for action table

import pandas as pd
import numpy as np


actions=[]
for i1 in [1]:
    for i2 in [0.1075,0.2366]:
        for i3 in [0.3159,0.6508]:
            for i4 in [0.3523,0.1894]:
                for i5 in [1]:
                    for i6 in [0.4303,0.1687]:
                        actions.append([i1,i2,i3,i4,i5,i6])


d=pd.DataFrame(actions)
writer = pd.ExcelWriter('./action_table_of_DQN3.xlsx')
d.to_excel(writer)
writer.save()

actions=pd.read_excel('./action_table_of_DQN3.xlsx').values[:,:6]
print(actions.shape)

'''
actions=[]

for i1 in [1]:
    for i2 in [0,1]:
        for i3 in [0,1]:
            for i4 in [0,1]:
                for i5 in [0,1]:
                    for i6 in [0,1]:
                        actions.append([i1,i2,i3,i4,i5,i6])


d=pd.DataFrame(actions)
writer = pd.ExcelWriter('./action_table_of_DQN_2.xlsx')
d.to_excel(writer)
writer.save()

actions=pd.read_excel('./action_table_of_DQN_2.xlsx').values[:,:6]
print(actions.shape)
'''