# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:03:01 2020

@author: Administrator
"""

from pyswmm import Simulation
import set_datetime
import pandas as pd
import numpy as np

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass  
    
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
        store=0
        outflow=0
        upflow=0
        downflow=0
        pumps_flag = outfall_flag= pumps_flag_cod= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            pumps_flag_cod = handle_line(line, pumps_flag_cod, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[] and not pumps_flag_cod:
                if line.find('External Outflow')>=0 or\
                   line.find('Exfiltration Loss')>=0 or \
                   line.find('Mass Reacted')>=0:
                    outflow+=float(node[3])

                elif line.find('Flooding Loss')>=0:
                    flooding=float(node[4])
                    pumps_flag=False
                    #print(flooding)
                elif line.find('Final Stored Mass')>=0:
                    store=float(node[4])
                elif line.find('Dry Weather Inflow')>=0 or \
                     line.find('Wet Weather Inflow')>=0 :
                    total_in+=float(node[4])
                    
                elif line.find('Groundwater Inflow')>=0 or \
                     line.find('RDII Inflow')>=0 or \
                     line.find('External Inflow')>=0:
                    total_in+=float(node[3])
                    
            if outfall_flag and node!=[]:
                if line.find('outfall-27')>=0 or line.find('outfall-28')>=0 or line.find('outfall-24')>=0:
                    upflow+=float(node[5])
                elif line.find('outfall-')>=0 or line.find('12')>=0 or line.find('67')>=0:
                    downflow+=float(node[5])
                    

    return total_in,flooding,store,outflow,upflow,downflow

date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50'
               ]

    
date_t=[]
for i in range(len(date_time)):
    date_t.append(int(i*10))
    

sdate=edate='08/28/2015'
stime=date_time[0]
etime=date_time[1]


writer = pd.ExcelWriter('./Final_results.xlsx')
names=['voting','hc','opt','dqn','ddqn','ppo1','ppo2','a2c']

floods_results={}

for name in names:
    floods=[]
    if name=='a2c':
        file_names=['./final-noAFI-drain/a2c_test_result/0','./final-noAFI-drain/a2c_test_result/1',
                    './final-noAFI-drain/a2c_test_result/2','./final-noAFI-drain/a2c_test_result/3']
    elif name=='dqn':
        file_names=['./final-noAFI-drain/dqn_test_result/0','./final-noAFI-drain/dqn_test_result/1',
                    './final-noAFI-drain/dqn_test_result/2','./final-noAFI-drain/dqn_test_result/3']
    elif name=='ddqn':
        file_names=['./final-noAFI-drain/ddqn_test_result/0','./final-noAFI-drain/ddqn_test_result/1',
                    './final-noAFI-drain/ddqn_test_result/2','./final-noAFI-drain/ddqn_test_result/3']
    elif name=='ppo1':
        file_names=['./final-noAFI-drain/ppo1_test_result/0','./final-noAFI-drain/ppo1_test_result/1',
                    './final-noAFI-drain/ppo1_test_result/2','./final-noAFI-drain/ppo1_test_result/3']
    elif name=='ppo2':
        file_names=['./final-noAFI-drain/ppo2_test_result/0','./final-noAFI-drain/ppo2_test_result/1',
                    './final-noAFI-drain/ppo2_test_result/2','./final-noAFI-drain/ppo2_test_result/3']
    elif name=='voting':
        file_names=['./final-noAFI-drain/voting_test_result/0','./final-noAFI-drain/voting_test_result/1',
                    './final-noAFI-drain/voting_test_result/2','./final-noAFI-drain/voting_test_result/3']
    elif name=='opt':
        file_names=['./final-noAFI-drain/opt_test_result/0','./final-noAFI-drain/opt_test_result/1',
                    './final-noAFI-drain/opt_test_result/2','./final-noAFI-drain/opt_test_result/3']
    else:
        file_names=['./final-noAFI-drain/HC/HC0','./final-noAFI-drain/HC/HC1',
                    './final-noAFI-drain/HC/HC2','./final-noAFI-drain/HC/HC3']


    for file_name in file_names:
        tem_floods=[]
        for iten in range(1,len(date_time)):
            tem_etime=date_time[iten]
            set_datetime.set_date(sdate,edate,stime,tem_etime,file_name+'.inp')
            simulation(file_name+'.inp')
            total_in,flooding,store,outflow,upflow,downflow=get_rpt(file_name+'.rpt')
            print(tem_etime,': ',flooding)
            tem_floods.append(flooding)
        floods.append(tem_floods)
    
    floods_results[name]=floods
    
    df = pd.DataFrame(np.array(floods).T)
    df.to_excel(writer, index=False, encoding='utf-8',sheet_name=name)
writer.save()
    
print(floods_results)

