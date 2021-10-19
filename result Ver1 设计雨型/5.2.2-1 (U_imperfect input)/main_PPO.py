# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:02:42 2019

@author: chong
"""

import PPO
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
    
    for AFI in [True,False]:
        if AFI:
            t='ppo2 safe'
        else:
            t='ppo2 nosafe'
        
        env=env_SWMM.env_SWMM(date_time, date_t,AFI)
        #observation=env.reset()
        #print(env.action_space)
        '''
        print(observation)
        for t in range(len(date_t)-3):
            r=env.step([0.3,0.7])
            print(env.iten)
            print(r)
        '''
        num_rain=10
        batch_size=240
        step=20
        test_num=10
        test_N=10
        
        random_level=[5,10,15,20]
        
        
        model2 = PPO.PPO(env,step, batch_size, num_rain, t)
        history = model2.train()
        for rainid in range(test_num):
            for rd in random_level:
                r2=model2.test_inp(test_N,rd,rainid)