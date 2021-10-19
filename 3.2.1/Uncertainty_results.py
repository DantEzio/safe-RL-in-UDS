# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:05:02 2021

@author: chong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from swmm_api import read_rpt_file

def handle_line(line, flag, title):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    return flag

def get_rpt(filename):
    with open(filename, 'rt') as data:
        total_in=0
        flooding=0
        pumps_flag = outfall_flag= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('Flooding Loss')>=0:
                    flooding=float(node[3])
                elif line.find('Dry Weather Inflow')>=0 or \
                     line.find('Wet Weather Inflow')>=0 :
                    total_in+=float(node[4])                  
                elif line.find('Groundwater Inflow')>=0 or \
                     line.find('RDII Inflow')>=0 or \
                     line.find('External Inflow')>=0:
                    total_in+=float(node[3])
    return total_in,flooding


def section_3_2_1():
    #获取文件名
    file_name=['./ddqn nosafe_test_result/','./ddqn safe_test_result/',
               './ppo2 nosafe_test_result/','./ppo2 safe_test_result/',
               './HC/','./BC_test_result/','./EFD_test_result/']
    
    savedata=[]
    for name in file_name:  
        tem=[]
        all_rate=[]
        for i in range(160):
            if name=='./HC/':
                f=name+'HC'+str(i)+'.rpt'
            else:
                f=name+str(i)+'.rpt'
            total_in,CSO=get_rpt(f)#读取total_inflow和CSO
            all_rate.append(CSO/total_in)#计算比值RIC
             
        
        #找到5%，50%，95%的结果
        all_rate.sort()
        print(name+':')
        savedata.append([all_rate[int(160*5/100)],
                         all_rate[int(160*50/100)],
                         all_rate[int(160*95/100)]])
        print('5%:',all_rate[int(160*5/100)],' ',
              '50%:',all_rate[int(160*50/100)],' ',
              '95%:',all_rate[int(160*95/100)])
        
    pd.DataFrame(savedata).to_excel('RCI.xlsx')
    
def sectcion_4_3():
    #获取文件名
    file_name=['./ddqn nosafe_test_result/','./ddqn safe_test_result/',
               './ppo2 nosafe_test_result/','./ppo2 safe_test_result/',
               './BC_test_result/','./EFD_test_result/']
    title=['DQN','Safe-DQN','PPO','Safe-PPO','BC','EFD']

    savedata=[]
    for name in file_name:  
        all_rate={'RCI':[],'volume':[]}
        for i in range(160):
            if name=='./HC/':
                f=name+'HC'+str(i)+'.rpt'
            else:
                f=name+str(i)+'.rpt'
            #get rainfall
            rpt = read_rpt_file(f)  # type: swmm_api.SwmmReport
            data = rpt.runoff_quantity_continuity  # type: pandas.DataFrame
            all_rate['volume'].append(data['Total Precipitation']['Depth_mm'])
            #get RCI
            total_in,CSO=get_rpt(f)#读取total_inflow和CSO
            all_rate['RCI'].append(CSO/total_in)#计算比值RIC
        #turn to dataframe for sort()
        all_rate=pd.DataFrame(all_rate)
        #print(all_rate)
        #all_rate.sort_index(axis=0,by='volume')
        savedata.append(all_rate)
    
    
    font0 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 40,
            }
    #draw scatter
    fig=plt.figure(figsize=(30,25))
    img=0
    for img in range(1,7):
        fig.add_subplot(3,2,img)
        plt.scatter(savedata[img-1]['volume'],savedata[img-1]['RCI'],s=30)
        plt.title(title[img-1],fontdict=font0)
        
        
        plt.yticks(fontsize=20)
        plt.ylabel('RCI',font0)
        if img>=5:
            plt.xlabel('Volume (mm)',font0)
            plt.xticks(fontsize=20)
        else:
            plt.xlabel('')
            plt.xticks(fontsize=20)
    fig.savefig('RCI_vs_volume_4_3.png',bbox_inches='tight',dpi=200)

def test_rpt():
    rpt = read_rpt_file('./ddqn safe_test_result/0.rpt')  # type: swmm_api.SwmmReport
    data = rpt.runoff_quantity_continuity  # type: pandas.DataFrame
    print(data['Total Precipitation']['Depth_mm'])
           
if __name__=='__main__':
    #section_3_2_1()
    sectcion_4_3()
    