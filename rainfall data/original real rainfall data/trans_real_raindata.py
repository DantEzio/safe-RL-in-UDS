# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:24:31 2021

@author: chong
"""

import numpy as np
import pandas as pd


def get_raindata():
    n_raingages=4
    
    raindatas = ['./{0}Astlingen_Erft{1}.txt'.format(i+1,i+1) for i in range(n_raingages)]
    files = pd.concat([pd.read_csv(raindatas[i],header=None,names=['date','time','RG%s'%(i+1)],sep='\s+') for i in range(n_raingages)],axis=1)
    files = files.loc[:,~files.columns.duplicated()]
    date_sum = files.groupby("date").agg('sum').sum(axis=1)  # 计算每天的降雨量
    #print(date_sum.index.shape,date_sum.values[10])
    #print(files[files['date']==date_sum.index[0]])
    print(np.mean(date_sum.values),np.min(date_sum.values))
    
    
    
    
    date=date_sum[date_sum.values>40]#选出日降雨量大于40的所有降雨，用于训练和不确定性分析，保存4场降雨
    for gage in ['RG1','RG2','RG3','RG4']:
        data=[]
        for d in date.index:
            #tem=(files[files['date']==d]['RG1']+files[files['date']==d]['RG2']+files[files['date']==d]['RG3']+files[files['date']==d]['RG4'])/4
            tem=(files[files['date']==d][gage])
            data.append(tem.tolist())
        
        np.savetxt('A_real_raindata'+gage+'_40.txt', data)
    
    
    date=date_sum[date_sum.values>80]#选出日降雨量大于80的所有降雨，用于test，保存4场降雨
    for gage in ['RG1','RG2','RG3','RG4']:
        data=[]
        for d in date.index:
            #tem=(files[files['date']==d]['RG1']+files[files['date']==d]['RG2']+files[files['date']==d]['RG3']+files[files['date']==d]['RG4'])/4
            tem=(files[files['date']==d][gage])
            data.append(tem.tolist())
        
        np.savetxt('A_real_raindata'+gage+'_80.txt', data)
    
    
def generate_timelabel():
    #生成四场降雨需要的时间戳
    for gage in ['RG1','RG2','RG3','RG4']:
        tem=''
        with open('original label.txt','r') as f:
            for line in f.readlines():
                tem+=gage+line
        with open('label_'+gage+'.txt','w') as f:
            f.write(tem)



if __name__=='__main__':
    #get_raindata()
    generate_timelabel()