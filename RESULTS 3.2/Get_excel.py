# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:20:55 2021

@author: chong

从各个excel中读取最大最小值 存起来

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())
os.chdir('C:/Users/chong/Desktop/RESULTS 5.2')
#获取所有文件名
files=[]
for item in ['5.2.2-1 (U_imperfect input)','5.2.2-2 (U_noisy output)','5.2.2-3  (U_input and output)']: 
    for testdataid in ['40 results','80 results']:
        if testdataid=='40 results':
            test_rains=[3,5,6,7,9]
        else:
            test_rains=[1,4,5,7,9]
        for rainid in test_rains:
            for t in ['ddqn nosafe','ddqn safe','ppo2 nosafe','ppo2 safe']:
                for rd in [5,10,15,20]:
                    files.append('/'+item+'/'+testdataid+'/'
                                +t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv')
print(os.getcwd())
#读取所有数据并做处理
data={}
for f in files:
    df=pd.read_csv('C:/Users/chong/Desktop/RESULTS 5.2'+f).values
    dfmax=np.max(df,axis=1)
    dfmin=np.min(df,axis=1)
    for i in range(dfmax.shape[0]-1):
        if dfmax[i+1]<dfmax[i]:
            dfmax[i+1]=dfmax[i]
        if dfmin[i+1]<dfmin[i]:
            dfmin[i+1]=dfmin[i]
    data[f]=[dfmax,dfmin]


#画3图
#Fig11
font0 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 20,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 20,
}

fig=plt.figure(figsize=(60,80))
img=1
myt=['0','480']
myx=[0,46]

rainids={'40 results':[3,5,6,7,9],
         '80 results':[5,10,15,20]}

for item in ['5.2.2-1 (U_imperfect input)','5.2.2-2 (U_noisy output)','5.2.2-3  (U_input and output)']: 
    for i in range(1,11):
        for j in range(1,5):
            fig.add_subplot(10,4,img)
            rd=j*5
            if i<5:
                testdataid='40 results'
            else:
                testdataid='80 results'
            rainid=rainids[testdataid][np.mod(i,5)]
            dataid='/'+item+'/'+testdataid+'/ddqn nosafe_test_result/Rain'+str(rainid-1)+'_'+'randomlevel'+str(rd)+'_ddqn nosafeflooding_vs_t.csv'
            tmax,tmin=data[dataid][0],data[dataid][1]
            x=range(tmax.shape[0])
            plt.fill_between(x,tmax,tmin,color='blue',alpha=0.5)
            
            
            plt.title('Rain'+str(rainid)+' Amplification='+str(rd/10),fontdict=font0)
            
            plt.xticks(myx,myt,fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('Time (minutes)',font1)
            plt.ylabel('Flooding ($\mathregular{10^3}\mathregular{m^3}$)',font1)
            img+=1
            #plt.legend(prop=font1)
    fig.savefig(item+'.png',dpi=200)

