#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:23:07 2020

@author: chong
"""

import pandas as pd
import matplotlib.pyplot as plt


dfa2c=pd.read_excel('./Final_results.xlsx',sheet_name='a2c')
dfddqn=pd.read_excel('./Final_results.xlsx',sheet_name='ddqn')
dfdqn=pd.read_excel('./Final_results.xlsx',sheet_name='dqn')
dfppo1=pd.read_excel('./Final_results.xlsx',sheet_name='ppo1')
dfppo2=pd.read_excel('./Final_results.xlsx',sheet_name='ppo2')
dfvt=pd.read_excel('./Final_results.xlsx',sheet_name='voting')
dfhc=pd.read_excel('./Final_results.xlsx',sheet_name='hc')
dfop=pd.read_excel('./Final_results.xlsx',sheet_name='opt')

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 10,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

fig = plt.figure(figsize=(15,15))
for im in range(1,5):
    plts=fig.add_subplot(2,2,im)
    plt.plot(dfddqn[im-1],'r:',label='DDQN')
    plt.plot(dfdqn[im-1],'g:',label='DQN')
    plt.plot(dfppo1[im-1],'b:',label='PPO1')
    plt.plot(dfppo2[im-1],'c:',label='PPO2')
    plt.plot(dfa2c[im-1],'m:',label='A2C')
    plt.plot(dfhc[im-1],'k.-',label='Water level system')
    plt.plot(dfop[im-1],'k',label='Optimization model')
    
    plt.xlabel('time (minute)',font2)
    plt.ylabel('CSO volume (10$^{3}$ m$^{3}$)',font2)

    plt.legend(prop=font1)

fig.savefig('5.1.1.png', bbox_inches='tight', dpi=500)