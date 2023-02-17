# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:05:02 2021

@author: chong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

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


#获取文件名
file_name=['./ddqn nosafe_test_result/','./ddqn safe_test_result/',
           './ppo2 nosafe_test_result/','./ppo2 safe_test_result/',
           './EFD_test_result/','./BC_test_result/']

file_name=['./ddqn nosafe_test_result/HC/']

for name in file_name:  
    all_rate=[]
    for i in range(200):
        f=name+'HC'+str(i)+'.rpt'
        total_in,CSO=get_rpt(f)#读取total_inflow和CSO
        all_rate.append(CSO/total_in)#计算比值RIC
         
    
    #找到5%，50%，95%的结果
    all_rate.sort()
    print(name+':')
    print('5%:',all_rate[int(50*5/100)],' ',
          '50%:',all_rate[int(50*50/100)],' ',
          '95%:',all_rate[int(50*95/100)])