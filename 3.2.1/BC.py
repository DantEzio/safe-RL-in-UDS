# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 09:13:36 2021

@author: chong
"""

import numpy as np
import swmm_api as sa
import pandas as pd
import env_SWMM
import EFD_get_output
import change_rain
import set_datetime
import get_rpt
import set_pump

from pyswmm import Simulation

class BC:
    
    def __init__(self,test_num,env, date_time, date_t):
        self.t='BC'
        self.test_num=test_num
        self.env=env
        self.pool_list=['T1','T2','T3','T4','T5','T6']
        self.pool_d=[0,0,0,0,0,0]
        self.pumps=['V1','V2','V3','V4','V5','V6']
        
        self.orf='./sim/orf'#原始的inp文件，只含有管网信息
        self.orf_rain='./sim/orf_rain'#
        self.staf='./sim/staf'#用于sim的inp文件，在orf基础上修改了时间与降雨
        self.orftem='./sim/orf_tem'#最终模拟使用的file
        
        self.date_time=date_time
        self.date_t=date_t
        self.T=len(self.date_t)
        
        self.sdate=self.edate='08/28/2015'
        #先sim10min
        self.stime=date_time[0]
        self.etime=date_time[1]
        
        
        #self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')/2.2#读取训练降雨数据
        #print(self.rainData.shape)
        #self.testRainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')/2.2
        #self.rainnum,m=self.testRainData.shape
        #self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        #self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
        #降雨数据变为字典，4个元素对应4个rg
        rains={}
        for rg in ['RG1','RG2','RG3','RG4']:
            rains[rg]=np.loadtxt('./sim/A_real_raindata/A_real_raindata'+rg+'_40.txt',delimiter=' ')#读取训练降雨数据
        self.rainnum,m=rains['RG1'].shape
        
        self.testRainData=[]
        for i in range(self.rainnum):
            tem={}
            for rg in ['RG1','RG2','RG3','RG4']:
                tem[rg]=rains[rg][i]
            self.testRainData.append(tem)
        
        
    
    def simulation(self,filename):
        with Simulation(filename) as sim:
            #stand_reward=0
            for step in sim:
                pass    
    
    def BC_control(self,T):
        V=[1.0,0.0,0.0,0.0,0.0,0.0]
        
        #Rule BC
        if T[0]>1:
            V[1]=0.2366
            V[2]=0.6508
            V[3]=0.3523
            V[5]=0.4303

        return V
    
    def BC_reset(self,i):
        for rg in ['RG1','RG2','RG3','RG4']:
            change_rain.change_rain(self.testRainData[i][rg],self.orf_rain+'.inp',rg)#先修改self.orf_rain，再复制给staf
        change_rain.copy_result(self.staf+'.inp',self.orf_rain+'.inp')
        change_rain.copy_result(self.orftem+'.inp',self.orf_rain+'.inp')
        
        self.iten=1
        self.action_seq=[]
        
        tem_etime=self.date_time[self.iten]
        set_datetime.set_date(self.sdate,self.edate,self.stime,tem_etime,self.staf+'.inp')
        
        self.simulation(self.staf+'.inp')
         
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.staf+'.rpt')
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        self.pool_d=EFD_get_output.depth(self.staf+'.out',self.pool_list,self.date_t[self.iten]-self.iten)
        return flooding
        
    def BC_step(self,a):
        #修改statf的date，根据iten逐步向前
        #加入action
        #开始模拟，存储结果
        self.iten+=1
        action=a#self.action_table[a,:][0].tolist()
        
        #设置pump并模拟之后才有reward
        if len(action)!=6:
            print('wrong')
            
        self.action_seq.append(action)
        set_pump.set_pump(self.action_seq,self.date_time[1:self.iten],self.pumps,self.orftem+'.inp')
        tem_etime=self.date_time[self.iten]
        set_datetime.set_date(self.sdate,self.edate,self.stime,tem_etime,self.orftem+'.inp')
        #change_rain.copy_result(infile+'.inp',startfile+'.inp')
        
        #还原SWMM缓存inp
        change_rain.copy_result(self.staf+'.inp',self.orftem+'.inp')
        change_rain.copy_result(self.orftem+'.inp',self.orf_rain+'.inp')
        
        #step forward
        self.simulation(self.staf+'.inp')
        
        self.pool_d=EFD_get_output.depth(self.staf+'.out',self.pool_list,self.date_t[self.iten]-self.iten)
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.staf+'.rpt')
        if self.iten==self.T-2:
            done=True
        else:
            done=False

        return done,flooding    
    
    def BC_test(self):

        flooding_logs=[]
        for i in range(self.test_num):
            #acc_r = [0]
            print('test',i)
            
            flooding = self.BC_reset(i)
            flooding_log=[flooding]
            #print('obtest=',observation)
            while True:
                # if total_steps-MEMORY_SIZE > 9000: env.render()
                action = self.BC_control(self.pool_d)
                done, flooding = self.BC_step(action)
                #self.pool_d=EFD_get_output.depth(self.pool_list)
                flooding_log.append(flooding)
                if done:
                    break
                    #dr.append(acc_r)
            
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            
            #save RLC .inp and .rpt
            sout='./'+self.t+'_test_result/'+str(i)+'.rpt'
            sin=self.env.staf+'.rpt'
            self.env.copy_result(sout,sin)
            sout='./'+self.t+'_test_result/'+str(i)+'.inp'
            sin=self.env.staf+'.inp'
            self.env.copy_result(sout,sin)
        #保存所有降雨的flooding过程线
        df = pd.DataFrame(np.array(flooding_logs).T)
        df.to_csv('./BC_test_result/flooding_vs_t.csv', index=False, encoding='utf-8')
            

if __name__=='__main__':
    
    date_time=['00:00','00:10','00:20','00:30','00:40','00:50',\
               '01:00','01:10','01:20','01:30','01:40','01:50',\
               '02:00','02:10','02:20','02:30','02:40','02:50',\
               '03:00','03:10','03:20','03:30','03:40','03:50',\
               '04:00','04:10','04:20','04:30','04:40','04:50',\
               '05:00','05:10','05:20','05:30','05:40','05:50',\
               '06:00','06:10','06:20','06:30','06:40','06:50',\
               '07:00','07:10','07:20','07:30','07:40','07:50',\
               '08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00','16:10','16:20','16:30','16:40','16:50',\
               '17:00','17:10','17:20','17:30','17:40','17:50',\
               '18:00','18:10','18:20','18:30','18:40','18:50',\
               '19:00','19:10','19:20','19:30','19:40','19:50',\
               '20:00','20:10','20:20','20:30','20:40','20:50',\
               '21:00','21:10','21:20','21:30','21:40','21:50',\
               '22:00','22:10','22:20','22:30','22:40','22:50',\
               '23:00','23:10','23:20','23:30','23:40','23:50']
    
    date_t=[]
    for i in range(len(date_time)):
        date_t.append(int(i*10))
        
    AFI=False
    
    test_num=160
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    efd=BC(test_num,env,date_time, date_t)
    efd.BC_test()
    
    
