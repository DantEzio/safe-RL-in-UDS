# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:27:15 2020

@author: Administrator
"""

import A2C
import PPO
import DDQN
import env_SWMM
import numpy as np
import pandas as pd



class Voting():
    def __init__(self,MAX_EPISODE,MAX_EP_STEPS,env,num_rain,AFI):
        
        self.model_a2c=A2C.A2C(MAX_EPISODE,MAX_EP_STEPS,num_rain,env,AFI)
        #self.model_a2c.train()
        #self.model_a2c.test(0)
        print('a2c')
        
        self.model_ppo1 = PPO.PPO(env, MAX_EPISODE,MAX_EP_STEPS,num_rain, 'ppo1',AFI)
        #self.model_ppo1.train()
        #self.model_ppo1.test(0)
        print('ppo1')
        
        self.model_ppo2 = PPO.PPO(env, MAX_EPISODE, MAX_EP_STEPS,num_rain,'ppo2',AFI)
        #self.model_ppo2.train()
        #self.model_ppo2.test(0)
        print('ppo2')
        
        self.model_dqn = DDQN.DDQN(step=MAX_EPISODE,batch_size=MAX_EP_STEPS,num_rain=num_rain,env=env,t='dqn',AFI=AFI)
        #self.model_dqn.train()
        #self.model_dqn.test(0)
        print('dqn')
        
        self.model_ddqn = DDQN.DDQN(step=MAX_EPISODE,batch_size=MAX_EP_STEPS,num_rain=num_rain,env=env,t='ddqn',AFI=AFI)
        #self.model_ddqn.train()
        #self.model_ddqn.test(0)
        print('ddqn')
        
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
        self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
        self.rainnum,m=self.rainData.shape
        
        self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
    
         
    def test(self,test_num,env):

        dr=[]
        #AFI_env='./sim/AFI/afi_env'
        
        flooding_logs,hc_flooding_logs=[],[]
        select_log=[]
        for i in range(test_num):
            print('test'+str(i))
            
            observation,flooding = env.reset(self.testRainData[i])
            #准备optimization of AFI使用的inp文件
            #change_rain.copy_result(env.staf+'.inp',AFI_env+'.inp')
            #用于对比的HC,HC有RLC同步，共用一个iten计数器，
            #所以HC的reset要紧跟RLC的reset，HC的step要紧跟RLC的step，保证iten变量同步
            
            flooding_log=[flooding]
            selects=[]
            states, actions, rewards = [], [], []
            while True:
                a1 = self.model_ppo1.choose_action(observation)[0]
                a2 = self.model_ppo2.choose_action(observation)[0]
                a3 = self.model_dqn.choose_action(observation)
                a4 = self.model_ddqn.choose_action(observation)
                a5 = self.model_a2c.actor.choose_action(observation)
                
                atem=[a1,a2,a3,a4,a5]
                floodtem=[]
                for j in range(len(atem)):
                    _,_,_,_,flooding = env.step(atem[j],self.testRainData[i])
                    floodtem.append(flooding)
                    env.iten-=1
                    env.action_seq.pop()
                
                #print(i,floodtem)
                pick=floodtem.index(min(floodtem))
                if pick==0:
                    selects.append('ppo1')
                elif pick==1:
                    selects.append('ppo2')
                elif pick==2:
                    selects.append('dqn')
                elif pick==3:
                    selects.append('ddqn')
                else:
                    selects.append('a2c')
                a=atem[pick]
                next_observation, reward, done, _, flooding = env.step(a,self.testRainData[i])
                
                #对比HC,也记录HC每一步的flooding
                
                states.append(observation)
                actions.append(a)
                
                flooding_log.append(flooding)
                
                rewards.append(reward)

                observation = next_observation
                    
                if done:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.model_ppo1.discount_reward(states, rewards, next_observation)

                    states, actions, rewards = [], [], []
                    dr.append(d_reward)
                    
                    break
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            select_log.append(selects)

            #对比HC,也记录HC每一步的flooding
            #self.env.copy_result('./'+self.t+'_test_result/HC/HC'+str(i)+'.inp',self.env.orf_rain+'.inp')
            #tem_etime=self.env.date_time[1]
            #set_datetime.set_date(self.env.sdate,self.env.edate,self.env.stime,tem_etime,'./'+self.t+'_test_result/HC/HC'+str(i)+'.inp')
            #self.env.simulation('./'+self.t+'_test_result/HC/HC'+str(i)+'.inp')
            
            #save RLC .inp and .rpt
            sout='./voting_test_result/'+str(i)+'.rpt'
            sin=env.staf+'.rpt'
            env.copy_result(sout,sin)
            sout='./voting_test_result/'+str(i)+'.inp'
            sin=env.staf+'.inp'
            env.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)
        
        #保存所有降雨的flooding过程线
        df = pd.DataFrame(np.array(flooding_logs).T)
        df.to_csv('./voting_test_result/voting_flooding_vs_t.csv', index=False, encoding='utf-8')
        df = pd.DataFrame(np.array(hc_flooding_logs).T)
        df.to_csv('./voting_test_result/voting_hc_flooding_vs_t.csv', index=False, encoding='utf-8')
        df = pd.DataFrame(np.array(select_log).T)
        df.to_csv('./voting_test_result/voting_select_vs_t.csv', index=False, encoding='utf-8')
        return dr
    
if __name__ == '__main__':
    
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00']
    
    date_t=[]
    for i in range(len(date_time)):
        date_t.append(int(i*10))
    
    num_rain=2
    AFI=False
    
    
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    model=Voting(10,100,env,num_rain,AFI) 
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    testnum=4
    model.test(testnum,env)
