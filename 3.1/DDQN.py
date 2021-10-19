# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:45:18 2019

@author: Administrator

The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)
tf.compat.v1.set_random_seed(1)


class DDQN:
    def __init__(self,step=200,batch_size=24,num_rain=10,env=gym.make('Pendulum-v0').unwrapped,t='dqn'):
        
        #action与实际操作的对应表
        self.action_table=pd.read_excel('./action_table_of_DQN3.xlsx').values[:,1:]
        ACTION_SPACE = self.action_table.shape[0]#2**4 #table3#6个orifices，每个有三种取值，一共所有的可能取值范围
        print('table shape:',self.action_table.shape)
        
        n_features=5
        memory_size= 150000
        e_greedy_increment=0.1
        e_greedy=0.5
        reward_decay=0.9
        learning_rate=0.001
        replace_target_iter=10
        output_graph=False
        
        self.t=t
        
        self.n_actions = ACTION_SPACE
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.5 if e_greedy_increment is not None else self.epsilon_max
        self.num_rain=num_rain
        
        self.traing_step=step
        
        if self.t=='ddqn':
            self.dueling = True      # decide to use dueling DQN or not
        else:
            self.dueling = False     # decide to use dueling DQN or not

        self.learn_step_counter = 0
        #self.memory = np.zeros((self.memory_size, n_features*2+2))
        self.memory=[]
        self._build_net()
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]
        
        self.env=env
        
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []
        
        #降雨数据变为字典，4个元素对应4个rg
        #所有降雨在一个list中，每个元素为字典
        rains={}
        for rg in ['RG1','RG2','RG3','RG4']:
            rains[rg]=np.loadtxt('./sim/A_real_raindata/A_real_raindata'+rg+'_40.txt',delimiter=' ')#读取训练降雨数据
        m,n=rains['RG1'].shape
        print(m,n)
        self.rainData=[]
        for i in range(m):
            tem={}
            for rg in ['RG1','RG2','RG3','RG4']:
                tem[rg]=rains[rg][i]
            self.rainData.append(tem)
        #print(self.rainData.shape)
        '''
        testRainData1=np.loadtxt('./sim/testRainFile.txt',delimiter=',')/2.2#读取测试降雨数据       
        testRainData2=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*4#读取测试降雨数据
        self.rainnum,m=testRainData2.shape
        testRainData2=np.hstack((testRainData2,np.zeros((testRainData2.shape[0],m))))
        self.testRainData=np.concatenate((testRainData1,testRainData2),axis=0)
        print(self.testRainData.shape)
        self.rainData=np.concatenate((self.testRainData,self.rainData),axis=0)
        print(self.rainData.shape)
        self.rainnum,m=self.rainData.shape
        '''
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

        #add white noise
        #self.rainData=self.rainData+np.random.randn(self.rainData.shape[0],self.rainData.shape[1])
        #self.testRainData=self.testRainData+np.random.randn(self.testRainData.shape[0],self.testRainData.shape[1])
        #self.rainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')#读取测试降雨数据
        #self.testRainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')#读取测试降雨数据
        
        
    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.compat.v1.variable_scope('l1',reuse=tf.compat.v1.AUTO_REUSE):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.compat.v1.variable_scope('Value',reuse=tf.compat.v1.AUTO_REUSE):
                    w2 = tf.compat.v1.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.compat.v1.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.compat.v1.variable_scope('Advantage',reuse=tf.compat.v1.AUTO_REUSE):
                    w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.compat.v1.variable_scope('Q',reuse=tf.compat.v1.AUTO_REUSE):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.compat.v1.variable_scope('Q',reuse=tf.compat.v1.AUTO_REUSE):
                    w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.compat.v1.variable_scope('eval_net',reuse=tf.compat.v1.AUTO_REUSE):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 100, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.compat.v1.variable_scope('loss',reuse=tf.compat.v1.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.variable_scope('train',reuse=tf.compat.v1.AUTO_REUSE):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.compat.v1.variable_scope('target_net',reuse=tf.compat.v1.AUTO_REUSE):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a, r], s_))
        '''
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        #print('s=',s)
        #print('[a,r=]',a,r)
        #print('s_=',s_)
        '''
        
        self.memory.append(transition)
        self.memory_counter += 1
        

    def choose_action(self, observation):
        #observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        
        return action
    
    def action(self, observation):
        #observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
        action = np.argmax(actions_value)
        return action

    def learn(self,total_step):
        '''
        if self.learn_step_counter >=total_step/10:#% self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        '''
        self.sess.run(self.replace_target_op)
        
        #print(self.learn_step_counter % self.replace_target_iter)
        sample_index = np.random.choice(total_step, size=self.batch_size)
        batch_memory=[]
        #print(self.memory)
        #print(batch_memory[:, -self.n_features:])
        for i in sample_index:
            #print(self.memory[int(i)])
            batch_memory.append(list(self.memory[int(i)]))
        #batch_memory=list(batch_memory)
        batch_memory=np.array(batch_memory)
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:,-self.n_features:]}) # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:,:self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        #print(q_target[[1,2]])
        #q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target[batch_index] = np.reshape(reward + self.gamma * np.max(q_next, axis=1),(self.batch_size,1))
        print(q_target.shape)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
    def train(self):
        #现在将训练过程改为：一次性采样所有降雨（数量num_rain），
        #之后将所有降雨过程的state、reward、action用于训练
        history = {'episode': [], 'Episode_reward': []}
        saver=tf.compat.v1.train.Saver()
        saver.restore(self.sess,"./save/"+self.t+"/"+self.t+" model.ckpt")
        for j in range(self.traing_step):
            total_steps = 0
            for i in range(self.num_rain):
                print('training steps:',j)
                print('sampling number:',i)
                acc_r = [0] 
                observation, _ = self.env.reset(self.rainData[i])  
                episode_reward=0
                
                while True:
                    # if total_steps-MEMORY_SIZE > 9000: env.render()
                    #print('ob=',observation)
                    a = self.choose_action(observation)#action为概率
                    action = self.action_table[a,:].tolist()
                    #f_action = (action-(self.n_actions-1)/2)/((self.n_actions)/1)   # [-2 ~ 2] float actions
                    observation_, reward, done, info, _ = self.env.step(action,self.rainData[i])
                    #print(observation_, reward, done, info)
                    #reward /= 10      # normalize to a range of (-1, 0)
                    
                    episode_reward = reward+episode_reward*self.gamma
                    
                    acc_r.append(reward + acc_r[-1]*self.gamma)  # accumulated reward 这里reward是指每一步的reward，acc_r是每场降雨叠加的
            
                    self.store_transition(observation, action, reward, observation_)
         
                    observation = observation_
                    total_steps += 1
                
                    if done:
                        break
                
                if total_steps-self.memory_size > 15000:
                    break
            
                self.learn(total_steps)           
            sp=saver.save(self.sess,"./save/"+self.t+"/"+self.t+" model.ckpt")
            print("model saved:",sp)
            
            history['episode'].append(i)
            history['Episode_reward'].append(episode_reward)
            print('Episode: {} | Episode reward: {:.2f}'.format(j, episode_reward))
       
        return self.cost_his, acc_r,history
    
    def test(self,test_num,testdataid):

        #降雨数据变为字典，4个元素对应4个rg
        rains={}
        for rg in ['RG1','RG2','RG3','RG4']:
            rains[rg]=np.loadtxt('./sim/A_real_raindata/A_real_raindata'+rg+'_'+testdataid+'.txt',delimiter=' ')#读取训练降雨数据
        self.rainnum,m=rains['RG1'].shape
        
        self.testRainData=[]
        for i in range(self.rainnum):
            tem={}
            for rg in ['RG1','RG2','RG3','RG4']:
                tem[rg]=rains[rg][i]
            self.testRainData.append(tem)

        if testdataid=='40':
            savefile='40 results'
        else:
            savefile='80 results'
        
        saver=tf.compat.v1.train.Saver()
        saver.restore(self.sess,"./save/"+self.t+"/"+self.t+" model.ckpt")
        
        dr=[]
        flooding_logs,hc_flooding_logs=[],[]
        for i in range(test_num):
            #acc_r = [0]
            print('test',i)
            
            observation, flooding = self.env.reset(self.testRainData[i])
            
            #用于对比的HC,HC有RLC同步，共用一个iten计数器，
            #所以HC的reset要紧跟RLC的reset，HC的step要紧跟RLC的step，保证iten变量同步
            hc_name='./'+savefile+'/'+self.t+'_test_result/HC/'+str(i)
            self.env.copy_result(hc_name+'.inp',self.env.orf_rain+'.inp')
            hc_flooding = self.env.reset_HC(hc_name)
            
            flooding_log,hc_flooding_log=[flooding],[hc_flooding]
            
            #print('obtest=',observation)
            while True:
                # if total_steps-MEMORY_SIZE > 9000: env.render()
        
                a = self.action(observation)
                action = self.action_table[a,:].tolist()
                #print(action)
                #f_action = (action-(self.n_actions-1)/2)/((self.n_actions)/4)   # [-2 ~ 2] float actions
                observation_, reward, done, info, flooding = self.env.step(action,self.testRainData[i])
                #对比HC,也记录HC每一步的flooding
                _, hc_flooding = self.env.step_HC(hc_name)
                
                flooding_log.append(flooding)
                hc_flooding_log.append(hc_flooding)
                
                #reward /= 10      # normalize to a range of (-1, 0)
                #acc_r.append(reward + acc_r[-1])  # accumulated reward
     
                observation = observation_
                if done:
                    break
                    #dr.append(acc_r)
            
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            hc_flooding_logs.append(hc_flooding_log)
            
            #对比HC
            #change_rain.copy_result('./test_result/HC/compare_tem_HC'+str(i)+'.inp',self.env.orf_rain+'.inp')#还原
            #tem_etime=self.env.date_time[self.env.iten]
            #set_datetime.set_date(self.env.sdate,self.env.edate,self.env.stime,tem_etime,'./test_result/HC/compare_tem_HC'+str(i)+'.inp')
            #self.env.simulation('./test_result/HC/compare_tem_HC'+str(i)+'.inp')

            #save RLC .inp and .rpt
            sout='./'+savefile+'/'+self.t+'_test_result/'+str(i)+'.rpt'
            sin=self.env.staf+'.rpt'
            self.env.copy_result(sout,sin)
            sout='./'+savefile+'/'+self.t+'_test_result/'+str(i)+'.inp'
            sin=self.env.staf+'.inp'
            self.env.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)
        #保存所有降雨的flooding过程线
        df = pd.DataFrame(np.array(flooding_logs).T)
        df.to_csv('./'+savefile+'/'+self.t+'_test_result/'+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
        df = pd.DataFrame(np.array(hc_flooding_logs).T)
        df.to_csv('./'+savefile+'/'+self.t+'_test_result/'+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
        return dr
    
    def save_history(self, history, name):
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
    
    
if __name__=='__main__':
    
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    #env = gym.make('CartPole-v0')
    env.seed(1)
    MEMORY_SIZE = 3000
    ACTION_SPACE = 25
    
    #sess = tf.Session()
    with tf.variable_scope('natural'):
        natural_DQN = DDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, dueling=False)
    
    with tf.variable_scope('dueling'):
        dueling_DQN = DDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, dueling=True, output_graph=True)
    
    #sess.run(tf.global_variables_initializer())
    
    c_natural, r_natural = natural_DQN.train()
    c_dueling, r_dueling = dueling_DQN.train()
    
    plt.figure(1)
    plt.plot(np.array(c_natural), c='r', label='natural')
    plt.plot(np.array(c_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('cost')
    plt.xlabel('training steps')
    plt.grid()
    
    plt.figure(2)
    plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()
    
    plt.show()