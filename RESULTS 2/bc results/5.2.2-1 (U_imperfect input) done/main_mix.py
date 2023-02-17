# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:28:26 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:02:42 2019

@author: chong
"""

import PPO
import A2C
import DDQN
import env_SWMM


if __name__=='__main__':
    '''
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,240]
    '''
    
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
    
    env=env_SWMM.env_SWMM(date_time, date_t,True)
    batch_size=24
    step=5
    natural_DQN = DDQN.DDQN(step=step,batch_size=batch_size,env=env,t='dqn')
    natural_DQN.train()
    test_num=4
    r1=natural_DQN.test(test_num)
    print('dqn')
    
    env=env_SWMM.env_SWMM(date_time, date_t,True)
    model2 = PPO.PPO(env,5, 50, 'ppo1')
    #history = model2.train()
    test_num=4
    r=model2.test(test_num)
    print('ppo1')
    
    env=env_SWMM.env_SWMM(date_time, date_t,True)
    model3 = PPO.PPO(env,5, 50, 'ppo2')
    #history = model3.train()
    test_num=4
    r=model3.test(test_num)
    print('ppo2')
    
    #env=env_SWMM.env_SWMM(date_time, date_t, True)
    #MAX_EPISODE = 5#200场降雨
    #MAX_EP_STEPS = 5   # maximum time step in one episode
    #model=A2C.A2C(MAX_EPISODE,MAX_EP_STEPS,env)
    #history=model.train()
    #test_num=4
    #r=model.test(test_num)
    #print('a2c')
    '''
    
    
    dueling_DQN = DDQN.DDQN(step=step,batch_size=batch_size,env=env,t='ddqn')
    dueling_DQN.train()
    r2=dueling_DQN.test(test_num)
    print('ddqn')
    '''