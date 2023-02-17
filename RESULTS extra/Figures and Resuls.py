# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:04:21 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from swmm_api import read_inp_file, read_out_file,read_rpt_file
from pyswmm import Simulation
import get_output
import flooding_fig


def plot_flooding():
    items=['HC','BC_test_result','EFD_test_result',
           'ddqn nosafe_test_result','ddqn safe_test_result',
           'ppo2 nosafe_test_result','ppo2 safe_test_result']
    pack='final-noAFI-drain'
    rains=[str(i) for i in range(10)]
    
    #读取flooding excel数据
    data={}
    for rain in [str(i) for i in range(10)]:
        data[rain]=pd.read_excel('Flooding results.xlsx',sheet_name='rain'+rain).values[:,1:]
    
    titles=['Do nothing','BC','EFD','DQN','Safe-DQN','PPO','Safe-PPO']
    fig=plt.figure(figsize=(28,36))
    
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 18,
    }
    font0 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
    }
    myt=['8:00','16:00']
    myx=[0,46]
    img=1
    for i in range(1,6):
        for j in range(1,3):
            fig.add_subplot(5,2,img)
            k=0
            for i in range(len(items)):
                plt.title('Rain'+str(img),fontdict=font0)
                plt.plot(data[str(img-1)][k],label=titles[k])
                plt.xticks(myx,myt,fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Time (minutes)',font1)
                plt.ylabel('Flooding and CSO volume ($\mathregular{10^3}\mathregular{m^3}$)',font1)
                k=k+1
            img+=1
            plt.legend(prop=font1)
    fig.savefig('5.1.1.png',dpi=200)
    
def plot_DQN_flow(V):
    #读取流量excel数据
    detV3_data=pd.read_excel('V1_flow.xlsx',sheet_name='rain1').values[:,2:]
    
    titles=['DQN','Safe-DQN']
    #V='V1'
    
    fig=plt.figure(figsize=(28,36))
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 18,
    }
    font0 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
    }
    
    
    myt=['9:00','16:00']
    myx=[0,470]
    img=1
    for i in range(1,6):
        for j in range(1,3):
            fig.add_subplot(5,2,img)
            k=0
            detV3_data=pd.read_excel(V+'_flow.xlsx',sheet_name='rain'+str(img-1)).values[:,2:]
            for i in range(3,5):
                plt.title('Rain'+str(img),fontdict=font0)
                plt.plot(detV3_data[i,:],label=titles[k])
                plt.xticks(myx,myt,fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Time (minutes)',font1)
                plt.ylabel('Flow ($\mathregular{m^3}/s$)',font1)
                k=k+1
            img+=1
            plt.legend(prop=font1)
    fig.savefig('5.1.2-'+V+'_'+'DQN'+'_.png',dpi=200)
    
def plot_PPO_flow(V):
    #读取流量excel数据
    detV3_data=pd.read_excel('V1_flow.xlsx',sheet_name='rain1').values[:,2:]
    titles=['PPO','Safe-PPO']
    #V='V6'
    
    fig=plt.figure(figsize=(28,36))
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 18,
    }
    font0 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
    }
    
    
    myt=['0','480']
    myx=[0,470]
    img=1
    for i in range(1,6):
        for j in range(1,3):
            fig.add_subplot(5,2,img)
            k=0
            detV3_data=pd.read_excel(V+'_flow.xlsx',sheet_name='rain'+str(img-1)).values[:,2:]
            for i in range(5,7):
                plt.title('Rain'+str(img),fontdict=font0)
                plt.plot(detV3_data[i,:],label=titles[k])
                plt.xticks(myx,myt,fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Time (minutes)',font1)
                plt.ylabel('Flow ($\mathregular{m^3}/s$)',font1)
                k=k+1
            img+=1
            plt.legend(prop=font1)
    fig.savefig('5.1.2-'+V+'_'+'PPO'+'_.png',dpi=200)
    
    
if __name__=='__main__':
    #plot_flooding()
    for i in range(6):
        V='V'+str(i+1)
        plot_PPO_flow(V)
        plot_DQN_flow(V)