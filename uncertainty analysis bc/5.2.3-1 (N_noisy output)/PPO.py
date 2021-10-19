# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:50:57 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:30:53 2019

@author: chong

PPO algorithm
"""

#import gym
import numpy as np
import pandas as pd
import tensorflow as tf

#import change_rain as cr


class PPO:
    def __init__(self, env, ep, batch, num_rain, t='ppo1'):
        self.t = t
        self.ep = ep
        
        self.action_size=6#5台设备
        
        self.log = 'model/{}_log'.format(t)

        #self.env = gym.make('Pendulum-v0')
        self.env = env
        self.batch = self.env.T
        
        #self.bound = self.env.action_space.high[0]
        #print(env.action_space)
        self.bound_high = np.array([1 for _ in range(self.action_size)])#self.env.action_space[0]
        self.bound_low = np.array([0.1 for _ in range(self.action_size)])#self.env.action_space[1]

        self.gamma = 0.9
        self.A_LR = 0.001
        self.C_LR = 0.001
        self.A_UPDATE_STEPS = 50
        self.C_UPDATE_STEPS = 50

        # KL penalty, d_target、β for ppo1
        self.kl_target = 0.01
        self.lam = 0.5
        # ε for ppo2
        self.epsilon = 0.2
        self.num_rain=num_rain

        self.sess = tf.compat.v1.Session()
        self.build_model()
        
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')/2.2#读取训练降雨数据
        print(self.rainData.shape)
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
        
        self.testRainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')/2.2
        self.rainnum,m=self.testRainData.shape
        
        self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
        
        

    def _build_critic(self):
        """critic model.
        """
        with tf.compat.v1.variable_scope('critic',reuse=tf.compat.v1.AUTO_REUSE):
            x = tf.layers.dense(self.states, 100, tf.nn.relu,kernel_initializer=tf.zeros_initializer(),bias_initializer=tf.zeros_initializer())

            self.v = tf.layers.dense(x, 1,kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer())
            self.advantage = self.dr - self.v

    def _build_actor(self, name, trainable):
        """actor model.
        """
        with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
            x = tf.layers.dense(self.states, 100, tf.nn.relu, 
                                trainable=trainable,kernel_initializer=tf.zeros_initializer(), 
                                bias_initializer=tf.zeros_initializer())

            mu = (self.bound_high-self.bound_low) * tf.layers.dense(x, self.action_size, tf.nn.softmax, 
                                                                    trainable=trainable,
                                                                    kernel_initializer=tf.zeros_initializer(), 
                                                                    bias_initializer=tf.zeros_initializer())-self.bound_low
            sigma = tf.layers.dense(x, self.action_size, tf.nn.softplus, trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return norm_dist, params

    def build_model(self):
        """build model with ppo loss.
        """
        # inputs
        self.states = tf.compat.v1.placeholder(tf.float32, [None, 5], 'states')
        self.action = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], 'action')
        self.adv = tf.compat.v1.placeholder(tf.float32, [None, 1], 'advantage')
        self.dr = tf.compat.v1.placeholder(tf.float32, [None, 1], 'discounted_r')

        # build model
        self._build_critic()
        nd, pi_params = self._build_actor('actor', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_actor', trainable=False)

        # define ppo loss
        with tf.compat.v1.variable_scope('loss',reuse=tf.compat.v1.AUTO_REUSE):
            # critic loss
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # actor loss
            with tf.compat.v1.variable_scope('surrogate',reuse=tf.compat.v1.AUTO_REUSE):
                ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))
                surr = ratio * self.adv

            if self.t == 'ppo1':
                self.tflam = tf.compat.v1.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(old_nd, nd)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else: 
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.- self.epsilon, 1.+ self.epsilon) * self.adv))

        # define Optimizer
        with tf.compat.v1.variable_scope('optimize',reuse=tf.compat.v1.AUTO_REUSE):
            self.ctrain_op = tf.compat.v1.train.AdamOptimizer(self.C_LR).minimize(self.closs)
            self.atrain_op = tf.compat.v1.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        with tf.compat.v1.variable_scope('sample_action',reuse=tf.compat.v1.AUTO_REUSE):
            self.sample_op = tf.squeeze(nd.sample(self.action_size), axis=1)

        # update old actor
        with tf.compat.v1.variable_scope('update_old_actor',reuse=tf.compat.v1.AUTO_REUSE):
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        tf.compat.v1.summary.FileWriter(self.log, self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, state):
        """choice continuous action from normal distributions.

        Arguments:
            state: state.

        Returns:
           action.
        """
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.states: state})[0]
        #对输出的action做限制
        if action[1]<=0.5:
            action[1]=0.1075
        else:
            action[1]=0.2366
            
        if action[2]<=0.5:
            action[2]=0.3159
        else:
            action[2]=0.6508
            
        if action[3]<=0.5:
            action[3]=0.3523
        else:
            action[3]=0.1894
        
        if action[5]<=0.5:
            action[5]=0.4303
        else:
            action[5]=0.1687
        
        #return np.clip(action, -self.bound, self.bound)
        return np.clip(action, self.bound_low, self.bound_high)

    def get_value(self, state):
        """get q value.

        Arguments:
            state: state.

        Returns:
           q_value.
        """
        if state.ndim < 2: state = state[np.newaxis, :]

        return self.sess.run(self.v, {self.states: state})

    def discount_reward(self, states, rewards, next_observation):
        """Compute target value.

        Arguments:
            states: state in episode.
            rewards: reward in episode.
            next_observation: state of last action.

        Returns:
            targets: q targets.
        """
        n=len(states[0])
        s = np.vstack([states, next_observation.reshape(-1, n)])
        q_values = self.get_value(s).flatten()

        targets = rewards + self.gamma * q_values[1:]
        targets = targets.reshape(-1, 1)

        return targets

# not work.
#    def neglogp(self, mean, std, x):
#        """Gaussian likelihood
#        """
#        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
#               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#               + tf.reduce_sum(tf.log(std), axis=-1)

    def update(self, states, action, dr):
        """update model.

        Arguments:
            states: states.
            action: action of states.
            dr: discount reward of action.
        """
        self.sess.run(self.update_old_actor)

        adv = self.sess.run(self.advantage,
                            {self.states: states,
                             self.dr: dr})
        #print('adv:',adv.shape)
        #print('s:',states.shape)
        #print('a:',action.shape)
        #print('dr:',dr.shape)

        # update actor
        if self.t == 'ppo1':
            # run ppo1 loss
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.states: states,
                     self.action: action,
                     self.adv: adv,
                     self.tflam: self.lam})

            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            # run ppo2 loss
            for _ in range(self.A_UPDATE_STEPS):
                self.sess.run(self.atrain_op,
                              {self.states: states,
                               self.action: action,
                               self.adv: adv})

        # update critic
        for _ in range(self.C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op,
                          {self.states: states,
                           self.dr: dr})

    def train(self):
        """train method.
        """
        #tf.compat.v1.reset_default_graph()
        #现在将训练过程改为：一次性采样所有降雨（数量num_rain），
        #之后将所有降雨过程的state、reward、action用于训练
        saver=tf.compat.v1.train.Saver()
        #saver.restore(self.sess,"./save/"+self.t+"/"+self.t+"model.ckpt")
        history = {'episode': [], 'Episode_reward': []}
        for j in range(self.ep):
            for i in range(self.num_rain):
                print('steps: ',j,' rain:',i)
                observation,_ = self.env.reset(self.rainData[i])
                states, actions, rewards = [], [], []
                episode_reward = 0
                while True:
                    action = self.choose_action(observation)
                    #action[0] = 1
                    action[0] = 1
                    action[4] = 1
                    #print(action)
                    next_observation, reward, done, _, _ = self.env.step(action,self.rainData[i])
                    states.append(observation)
                    actions.append(action)
    
                    episode_reward += reward+episode_reward*self.gamma
                    rewards.append(reward)
    
                    observation = next_observation
                    
                    if done:
                        #dr=self.discount_reward(states, rewards, next_observation)
                        #print(sum(dr)[0])
                        #episode_reward +=sum(dr)[0]
                        break
                    
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            d_reward = self.discount_reward(states, rewards, next_observation)
            self.update(states, actions, d_reward)
            
            #保存模型
            sp=saver.save(self.sess,"./save/"+self.t+"/"+self.t+"model.ckpt")
            print("model saved:",sp)
            
            history['episode'].append(j)
            history['Episode_reward'].append(episode_reward)
            print('Episode: {} | Episode reward: {:.2f}'.format(j, episode_reward))
            
        return history
    
    def test(self,test_num):
        """train method.
        """
        saver=tf.compat.v1.train.Saver()
        saver.restore(self.sess,"./save/"+self.t+"/"+self.t+"model.ckpt")

        dr=[]
        
        flooding_logs,hc_flooding_logs=[],[]
        for i in range(test_num):
            print('test'+str(i))
            
            observation,flooding = self.env.reset(self.testRainData[i])
            #用于对比的HC,HC有RLC同步，共用一个iten计数器，
            #所以HC的reset要紧跟RLC的reset，HC的step要紧跟RLC的step，保证iten变量同步
            hc_name='./'+self.t+'_test_result/HC/HC'+str(i)
            self.env.copy_result(hc_name+'.inp',self.env.orf_rain+'.inp')
            hc_flooding = self.env.reset_HC(hc_name)           
            flooding_log,hc_flooding_log=[flooding],[hc_flooding]           
            states, actions, rewards = [], [], []
            
            while True:
                action = self.choose_action(observation)
                #print(a)
                action[0] = 1
                action[4] = 1
                print(action)
                next_observation, reward, done, _, flooding = self.env.step(action,self.testRainData[i])
                #对比HC,也记录HC每一步的flooding
                _, hc_flooding = self.env.step_HC(hc_name)
                
                states.append(observation)
                actions.append(action)
                
                flooding_log.append(flooding)
                hc_flooding_log.append(hc_flooding)
                
                rewards.append(reward)

                observation = next_observation
                    
                if done:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.discount_reward(states, rewards, next_observation)

                    #self.update(states, actions, d_reward)

                    states, actions, rewards = [], [], []
                    dr.append(d_reward)
                    
                    break
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            hc_flooding_logs.append(hc_flooding_log)
            
            #对比HC,也记录HC每一步的flooding
            #self.env.copy_result('./'+self.t+'_test_result/HC/HC'+str(i)+'.inp',self.env.orf_rain+'.inp')
            #tem_etime=self.env.date_time[1]
            #set_datetime.set_date(self.env.sdate,self.env.edate,self.env.stime,tem_etime,'./'+self.t+'_test_result/HC/HC'+str(i)+'.inp')
            #self.env.simulation('./'+self.t+'_test_result/HC/HC'+str(i)+'.inp')
            
            #save RLC .inp and .rpt
            sout='./'+self.t+'_test_result/'+str(i)+'.rpt'
            sin=self.env.staf+'.rpt'
            self.env.copy_result(sout,sin)
            sout='./'+self.t+'_test_result/'+str(i)+'.inp'
            sin=self.env.staf+'.inp'
            self.env.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)
        
        #保存所有降雨的flooding过程线
        df = pd.DataFrame(np.array(flooding_logs).T)
        df.to_csv('./'+self.t+'_test_result/'+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
        df = pd.DataFrame(np.array(hc_flooding_logs).T)
        df.to_csv('./'+self.t+'_test_result/'+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
        return dr
    
    def test_inp(self,test_num,rd,rainid):
        
        #saver=tf.compat.v1.train.Saver()
        #saver.restore(self.sess,"./save/"+self.t+"/"+self.t+" model.ckpt")
        
        dr=[]
        #flooding_logs,hc_flooding_logs=[],[]
        flooding_logs=[]
        #testid是指降雨编号
        #rainid是指重复测试编号
        for testid in range(test_num):
            i=rainid
            #acc_r = [0]
            print('test:',testid,' rain number:',rainid,' random level:',rd)
            
            observation, flooding = self.env.reset(self.testRainData[i])
            #uncertainty
            #用于对比的HC,HC有RLC同步，共用一个iten计数器，
            #所以HC的reset要紧跟RLC的reset，HC的step要紧跟RLC的step，保证iten变量同步
            #hc_name='./'+self.t+'_test_result/HC/HC'+str(i)
            #self.env.copy_result(hc_name+'.inp',self.env.orf_rain+'.inp')
            #hc_flooding = self.env.reset_HC(hc_name)
            
            #flooding_log,hc_flooding_log=[flooding],[hc_flooding]
            flooding_log=[flooding]
            #print('obtest=',observation)
            while True:
                # if total_steps-MEMORY_SIZE > 9000: env.render()
                
                #uncertainty
                #X=np.random.uniform(0.95,1.05,size=observation.shape)
                #observation=observation*X
                #observation=observation*X*rd
                #observation=observation*X
                
                action = self.choose_action(observation)
                X=np.random.normal(rd/10,0.01,size=action.shape)
                action=action+0.1*X
                #print(a)
                action[0] = 1
                action[4] = 1
                for it in range(len(action)):
                    if action[it]>1:
                        action[it]=1
                    elif action[it]<0:
                        action[it]=0
        
                #f_action = (action-(self.n_actions-1)/2)/((self.n_actions)/4)   # [-2 ~ 2] float actions
                observation_, reward, done, info, flooding = self.env.step(action,self.testRainData[i])
                
                #对比HC,也记录HC每一步的flooding
                #_, hc_flooding = self.env.step_HC(hc_name)
                
                flooding_log.append(flooding)
                #hc_flooding_log.append(hc_flooding)
                
                #reward /= 10      # normalize to a range of (-1, 0)
                #acc_r.append(reward + acc_r[-1])  # accumulated reward
     
                observation = observation_
                if done:
                    break
                    #dr.append(acc_r)
            
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            #hc_flooding_logs.append(hc_flooding_log)
            
            #对比HC
            #change_rain.copy_result('./test_result/HC/compare_tem_HC'+str(i)+'.inp',self.env.orf_rain+'.inp')#还原
            #tem_etime=self.env.date_time[self.env.iten]
            #set_datetime.set_date(self.env.sdate,self.env.edate,self.env.stime,tem_etime,'./test_result/HC/compare_tem_HC'+str(i)+'.inp')
            #self.env.simulation('./test_result/HC/compare_tem_HC'+str(i)+'.inp')

            #save RLC .inp and .rpt
            #sout='./'+self.t+'_test_result/'+str(testid)+'_'+str(rainid)+'.rpt'
            #sin=self.env.staf+'.rpt'
            #self.env.copy_result(sout,sin)
            #sout='./'+self.t+'_test_result/'+str(testid)+'_'+str(rainid)+'.inp'
            #sin=self.env.staf+'.inp'
            #self.env.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)
        #保存所有降雨的flooding过程线
        df = pd.DataFrame(np.array(flooding_logs).T)
        df.to_csv('./'+self.t+'_test_result/Rain'+str(rainid)+'_'+'randomlevel'+str(rd)+'_'+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
        #df = pd.DataFrame(np.array(hc_flooding_logs).T)
        #df.to_csv('./'+self.t+'_test_result/'+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
        return dr
    
    
    def save_history(self, history, name):
        #name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')


if __name__ == '__main__':
    '''
    model1 = PPO(1000, 32, 'ppo1')
    history = model1.train()
    model1.save_history(history, 'ppo1.csv')
    '''
    model2 = PPO(1000, 32, 'ppo2')
    history = model2.train()
    model2.save_history(history, 'ppo2.csv')