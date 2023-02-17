# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:20:55 2021

@author: chong

从各个excel中读取最大最小值 存起来

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



def plot_barh(item):
    #用于画maxf-minf的bar图
    
    
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
    
    #读取所有数据并做处理
    data={}
    for f in files:
        df=pd.read_csv(f).values[:,:10]
        dfmax=np.max(df,axis=1)
        dfmin=np.min(df,axis=1)
        for i in range(dfmax.shape[0]-1):
            if dfmax[i+1]<dfmax[i]:
                dfmax[i+1]=dfmax[i]
            if dfmin[i+1]<dfmin[i]:
                dfmin[i+1]=dfmin[i]
        data[f]=[dfmax,dfmin]

    #Fig
    font0 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 40,
    }
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 40,
    }
    
    
    #item='5.2.2-2 (U_noisy output)'
    ts=['ddqn nosafe','ddqn safe','ppo2 nosafe','ppo2 safe']
    
    wieght=0.8
    rag=5*2+wieght
    
    for t in ts:
        fig=plt.figure(figsize=(30,40))
        for rainid in range(10):
              
            rd=5
            f='./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
            tem=data[f][0][-1]-data[f][1][-1]
            if rainid==0:
                plt.barh(rainid*rag,tem,wieght,color='b',label='$\delta$=0.5')
            else:
                plt.barh(rainid*rag+1,tem,wieght,color='b')
            #plt.text(rainid*rag,tem+2,'$\delta$',fontsize=20)
            
            rd=10
            f='./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
            tem=data[f][0][-1]-data[f][1][-1]
            if rainid==0:
                plt.barh(rainid*rag+2,tem,wieght,color='y',label='$\delta$=1.0')
            else:
                plt.barh(rainid*rag+3,tem,wieght,color='y')
            #plt.text(rainid*rag+1,tem+2,'$\delta$',fontsize=20)
            
            rd=15
            f='./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
            tem=data[f][0][-1]-data[f][1][-1]
            if rainid==0:
                plt.barh(rainid*rag+4,tem,wieght,color='g',label='$\delta$=1.5')
            else:
                plt.barh(rainid*rag+5,tem,wieght,color='g')
            #plt.text(rainid*rag+2,tem+2,'$\delta$',fontsize=20)
            
            rd=20
            f='./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
            tem=data[f][0][-1]-data[f][1][-1]
            if rainid==0:
                plt.barh(rainid*rag+6,tem,wieght,color='r',label='$\delta$=2.0')
            else:
                plt.barh(rainid*rag+7,tem,wieght,color='r')
            #plt.text(rainid*rag+3,tem+2,'$\delta$',fontsize=20)
        
        plt.xlabel('$max_{f}-min_{f}$ ($\mathregular{10^3}\mathregular{m^3}$)',font1)
        plt.legend(fontsize=40)
        myx=[rag*i+3 for i in range(10)]
        myt=['Rain'+str(i) for i in range(1,11)]
        plt.yticks(myx,myt,fontsize=40)
        plt.xticks(fontsize=40)
        
        fig.savefig(item+t+'.png',dpi=400)

def plot_boxh():
    
    #用于画maxf-minf的bar图
    #Fig
    font0 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 40,
    }
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 40,
    }
    
    item='5.2.2-3 (U_input and output)'
    ts=['ddqn nosafe','ddqn safe','ppo2 nosafe','ppo2 safe']
    
    wieght=0.8
    rag=5*2+wieght
    
    for t in ts:
        
        files=[]
        for rainid in range(10):
            for rd in [5,10,15,20]:
                files.append('./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv')

        
        #读取所有t对应数据并做处理
        data=[]
        x=[]
        k=0
        for f in files:
            df=pd.read_csv(f).values
            dd=df[-1,:]
            data.append(dd)
            x.append(k*5)
        data=np.array(data)
        
        fig=plt.figure(figsize=(30,40))
        for rainid in range(10):
            plt.boxplot(data.T,vert=False,patch_artist=True)
            
        
        plt.xlabel('Flooding ($\mathregular{10^3}\mathregular{m^3}$)',font1)
        #plt.legend(fontsize=40)
        #myx=[rag*i+3 for i in range(10)]
        #myt=['Rain'+str(i) for i in range(1,11)]
        #plt.yticks(myx,myt,fontsize=40)
        #plt.xticks(fontsize=40)
        
        fig.savefig(item+t+'box.png',dpi=400)

def static_data():
    #用于计算每个的average flooding
    item='5.2.2-1 (U_imperfect input)'
    ts=['ddqn nosafe','ddqn safe','ppo2 nosafe','ppo2 safe']
    print(item)
    for t in ts:
        print(t)
        data=[]
        for rainid in range(10):
            tem=[]
            for rd in [5,10,15,20]:
                f='./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                df=pd.read_csv(f).values[:,:10]
                tem.append(np.round(np.mean(df[-1,:]),2))
            data.append(tem)
        data=pd.DataFrame(np.array(data).T)
        data.to_excel(item+'_'+t+'tem.xlsx')
            
def plot_box_combine(item,RLlist):
    #用于画maxf-minf的bar图
    if RLlist=='DQN':
        RL=['ddqn nosafe','ddqn safe']
    else:
        RL=['ppo2 nosafe','ppo2 safe']
    #获取所有文件名
    files=[]
    #for item in ['5.2.2-1 (U_imperfect input)','5.2.2-2 (U_noisy output)','5.2.2-3 (U_input and output)']: 
    for testdataid in ['40 results','80 results']:
        if testdataid=='40 results':
            test_rains=[3,5,6,7,9]
        else:
            test_rains=[1,4,5,7,9]
        for rainid in test_rains:
            for t in RL:
                for rd in [5,10,15,20]:
                    files.append('./'+item+'/'+testdataid+'/'
                                 +t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv')        
    #读取所有数据并做处理
    data={}
    for f in files:
        df=pd.read_csv(f).values[-1,:]
        data[f]=df

    #Fig
    font0 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 80,
            }
    
    font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 60,
            }

    k=1
    fig=plt.figure(figsize=(60,80))#10行1列
    for testdataid in ['40 results','80 results']:
        
        if testdataid=='40 results':
            test_rains=[3,5,6,7,9]
        else:
            test_rains=[1,4,5,7,9]
        for rainid in test_rains:
            #左边画DQN
            line=[]
            for t in RL:
                for rd in [5,10,15,20]:
                    #所有降雨，每场降雨一个子图，每个子图含有4个rd*4个RLs
                    f='./'+item+'/'+testdataid+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                    line.append(data[f].tolist())
                
            line=np.array(line).T            
            ax=fig.add_subplot(5,2,k)
            ax.boxplot(line, vert=True,patch_artist=True,showfliers=False)
            ax.set_title('Rain'+str(k),font0)
            ax.spines['top'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            ax.spines['right'].set_linewidth(5)
            
            if k>5*2-2:#最后一行
                ax.set_xticks([i for i in range(1,9)])
                ax.set_xticklabels([RLlist+'  \n'+'$\delta$=0.5  ',
                                    RLlist+'  \n'+'$\delta$=1.0  ',
                                    RLlist+'  \n'+'$\delta$=1.5  ',
                                    RLlist+'  \n'+'$\delta$=2.0  ',
                                    'Safe-'+RLlist+'  \n'+'$\delta$=0.5  ',
                                    'Safe-'+RLlist+'  \n'+'$\delta$=1.0  ',
                                    'Safe-'+RLlist+'  \n'+'$\delta$=1.5  ',
                                    'Safe-'+RLlist+'  \n'+'$\delta$=2.0  '],
                                   rotation=90,fontsize=60,
                                   fontname='Times New Roman')
            else:
                ax.set_xlabel('')
                ax.set_xticks([])
            
            lab='Flooding volume ($\mathregular{10^3}\mathregular{m^3}$)'
            ax.set_ylabel(lab,font1)
            ax.set_yticks([np.min(line),np.max(line)])
            ax.set_yticklabels([str(np.round(np.min(line),2)),
                                str(np.round(np.max(line),2))],
                               fontsize=60,
                               fontname='Times New Roman')
            
            k+=1
    fig.savefig(RLlist+' '+item+'.png',bbox_inches='tight',dpi=100)    
        



def plot_barh_combine():
    #用于画maxf-minf的bar图
    #print(os.getcwd())
    #获取所有文件名
    files=[]
    for item in ['5.2.2-1 (U_imperfect input)','5.2.2-2 (U_noisy output)','5.2.2-3 (U_input and output)']: 
        for testdataid in ['40 results','80 results']:
            if testdataid=='40 results':
                test_rains=[3,5,6,7,9]
            else:
                test_rains=[1,4,5,7,9]
            for rainid in test_rains:
                for t in ['ddqn nosafe','ddqn safe','ppo2 nosafe','ppo2 safe']:
                    for rd in [5,10,15,20]:
                        files.append('./'+item+'/'+testdataid+'/'
                                    +t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv')
        
    #读取所有数据并做处理
    data={}
    for f in files:
        df=pd.read_csv(f).values[:,:]
        dfmax=np.max(df,axis=1)
        dfmin=np.min(df,axis=1)
        for i in range(dfmax.shape[0]-1):
            if dfmax[i+1]<dfmax[i]:
                dfmax[i+1]=dfmax[i]
            if dfmin[i+1]<dfmin[i]:
                dfmin[i+1]=dfmin[i]
        data[f]=[dfmax,dfmin]

    #Fig
    font0 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 30,
    }
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 30,
    }
    
    
    item='5.2.2-2 (U_noisy output)'
    ts=['ddqn nosafe','ddqn safe','ppo2 nosafe','ppo2 safe']
    
    wieght=0.8
    rag=5*2
    
    for t in ts:
        
        items=['5.2.2-1 (U_imperfect input)',
              '5.2.2-2 (U_noisy output)',
              '5.2.2-3 (U_input and output)']
        fig,ax=plt.subplots(1,3,figsize=(30,20))
        k=0
        for item in items:
            for testdataid in ['40 results','80 results']:
            
                if testdataid=='40 results':
                    test_rains=[3,5,6,7,9]
                else:
                    test_rains=[1,4,5,7,9]
                
                
                for rainid in test_rains:
                    rd=5
                    f='./'+item+'/'+testdataid+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                    #'./'+item+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                    tem=data[f][0][-1]-data[f][1][-1]
                    if rainid==1:
                        ax[k].barh(rainid*rag,tem,wieght,color='b',label='$\delta$=0.5')
                    else:
                        ax[k].barh(rainid*rag+1,tem,wieght,color='b')
                    #plt.text(rainid*rag,tem+2,'$\delta$',fontsize=20)
                    
                    rd=10
                    f='./'+item+'/'+testdataid+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                    tem=data[f][0][-1]-data[f][1][-1]
                    if rainid==1:
                        ax[k].barh(rainid*rag+2,tem,wieght,color='y',label='$\delta$=1.0')
                    else:
                        ax[k].barh(rainid*rag+3,tem,wieght,color='y')
                    #plt.text(rainid*rag+1,tem+2,'$\delta$',fontsize=20)
                    
                    rd=15
                    f='./'+item+'/'+testdataid+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                    tem=data[f][0][-1]-data[f][1][-1]
                    if rainid==1:
                        ax[k].barh(rainid*rag+4,tem,wieght,color='g',label='$\delta$=1.5')
                    else:
                        ax[k].barh(rainid*rag+5,tem,wieght,color='g')
                    #plt.text(rainid*rag+2,tem+2,'$\delta$',fontsize=20)
                    
                    rd=20
                    f='./'+item+'/'+testdataid+'/'+t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+t+'flooding_vs_t.csv'
                    tem=data[f][0][-1]-data[f][1][-1]
                    if rainid==1:
                        ax[k].barh(rainid*rag+6,tem,wieght,color='r',label='$\delta$=2.0')
                    else:
                        ax[k].barh(rainid*rag+7,tem,wieght,color='r')
                    plt.text(rainid*rag+3,tem+2,'$\delta$',fontsize=20)
            ax[k].set_xlabel('$max_{f}-min_{f}$ ($\mathregular{10^3}\mathregular{m^3}$)',font1)
            ax[k].legend(fontsize=20)
            
            if k==0:
                ax[k].set_title('Imperfect input',fontproperties=font0)
            elif k==1:
                ax[k].set_title('Noisy output',fontproperties=font0)
            else:
                ax[k].set_title('Both',fontproperties=font0)
                       
            k=k+1
            
        myx=[rag*i+3 for i in range(10)]
        myt=['Rain'+str(i) for i in range(1,11)]
        plt.setp(ax, yticks=myx, yticklabels=myt)
        
        #fig.savefig(t+'.png',bbox_inches='tight',dpi=50)
    
    
if __name__=='__main__':
    #plot_barh_combine()
    for item in ['5.2.2-2 (U_noisy output)','5.2.2-3 (U_input and output)','5.2.2-1 (U_imperfect input)']:
        for RLlist in ['DQN','PPO']:
            plot_box_combine(item,RLlist)