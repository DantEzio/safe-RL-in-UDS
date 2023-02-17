# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:03:01 2020

@author: Administrator
"""

from pyswmm import Simulation
import pandas as pd
import numpy as np
import get_rpt

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass  


def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def set_date(sdate,edate,stime,etime,infile):
    temfile=infile+'tem_date.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            node=line.split()
            if node!=[]:
                if(node[0]=='START_DATE'):
                    tem=node[0]+' '*11+sdate
                    line=tem
                elif(node[0]=='END_DATE'):
                    tem=node[0]+' '*13+edate
                    line=tem
                elif(node[0]=='REPORT_START_DATE'):
                    tem=node[0]+' '*4+sdate
                    line=tem
                elif(node[0]=='REPORT_START_TIME'):
                    tem=node[0]+' '*4+stime
                    line=tem
                elif(node[0]=='START_TIME'):
                    tem=node[0]+' '*11+stime
                    line=tem
                elif(node[0]=='END_TIME'):
                    tem=node[0]+' '*13+etime
                    line=tem
    
                else:
                    pass
            else:
                pass
            output.write(line + '\n')
    output.close()
    copy_result(infile,temfile)
    
def Get_flooding(file_name):
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

    floods=[]
    for iten in range(1,len(date_time)):
        tem_etime=date_time[iten]
        set_date(sdate,edate,stime,tem_etime,file_name+'.inp')
        simulation(file_name+'.inp')
        flooding=get_rpt.get_rpt_flooding(file_name+'.rpt')
        floods.append(flooding)
    
    return floods

if __name__=='__main__':
    filename='./5.2.2-1 (U_imperfect input) done/BC_test_result/0'
    print(Get_flooding(filename))