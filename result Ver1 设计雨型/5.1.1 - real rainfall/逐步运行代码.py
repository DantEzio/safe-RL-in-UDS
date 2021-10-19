# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:54:10 2021

@author: MOMO
"""

from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api import read_inp_file,swmm5_run,read_rpt_file
import random
from math import log10
from os.path import exists
from numpy import std,average,argmax,array,logspace,multiply,identity
from networkx import Graph,adjacency_matrix
REWARD_DISCOUNT = 0.1

class Ast:
    def __init__(self,inp_file = './Astlingen.inp',
                 train_inp_file = './train/ep_%s.inp',
                 test_inp_file = './test/test.inp',
                 BC_inp = './BC.inp',
                 BC_inp_file = './test/test_BC.inp',
                 EFD_inp = './EFDC.inp',
                 EFD_inp_file = './test/testEFD.inp',
                 para_tuple = (0.95981,0.846,0.656,7,5,120),
                 rain_vari = (0.405,0.4,1,10)):
        self.n_agents = 6
        self.control_step = 300
        # self.duration = 2*3600
        # self.timesteps = self.duration/self.control_step+1
        self.action_size = 5
        self.observ_size = 4
        self.orfices = ['V'+str(i+1) for i in range(self.n_agents)]
        self.tanks = ['T'+str(i+1) for i in range(self.n_agents)]
        G = Graph([('V6','V3'),('V3','V1'),('V4','V1'),('V5','V1'),('V2','V1')])
        self.adj_matrix = adjacency_matrix(G,self.orfices).toaray()+identity(self.n_agents)
        
        self.train_inp_file = train_inp_file
        self.train_file = ''
        self.test_inp_file = test_inp_file
        self.BC_inp_file = BC_inp_file
        self.BC_inp = read_inp_file(BC_inp)
        self.EFD_inp_file = EFD_inp_file
        self.EFD_inp = read_inp_file(EFD_inp)
        self.inp = read_inp_file(inp_file)
        self.areas = [average([b for a,b in cur.points]) for cur in self.inp.CURVES.values()]
        self.para_tuple = para_tuple
        self.rain_vari = rain_vari
        self.disc = REWARD_DISCOUNT
        
        
    def generate_rain(self):
        A,C,n,b,delta,dura = self.para_tuple
        r,rd,Pa,Pb = self.rain_vari
        P = random.randint(Pa,Pb)
        r = random.gauss(r, rd)
        a = A*(1+C*log10(P))
        ts = []
        for i in range(dura//delta+1):
            t = i*delta
            key = str(t//60).zfill(2)+':'+str(t % 60).zfill(2)
            if t <= r*dura:
                ts.append([key, (a*((1-n)*(r*dura-t)/r+b)/((r*dura-t)/r+b)**(1+n))*60])
            else:
                ts.append([key, (a*((1-n)*(t-r*dura)/(1-r)+b)/((t-r*dura)/(1-r)+b)**(1+n))*60])
        return ts
    
    def Chicago_Hyetographs(self,r,P):
        A,C,n,b,delta,dura = self.para_tuple
        a = A*(1+C*log10(P))
        ts = []
        for i in range(dura//delta+1):
            t = i*delta
            key = str(t//60).zfill(2)+':'+str(t % 60).zfill(2)
            if t <= r*dura:
                ts.append([key, (a*((1-n)*(r*dura-t)/r+b)/((r*dura-t)/r+b)**(1+n))*60])
            else:
                ts.append([key, (a*((1-n)*(t-r*dura)/(1-r)+b)/((t-r*dura)/(1-r)+b)**(1+n))*60])
        return ts
    
    
    def run_simulation(self,agents,train=True):
        observs = []
        actions = []
        rewards = []
        states = []
        inp_file = self.train_file if train else self.test_inp_file
        total_rain = sum([b for _,b in read_inp_file(inp_file).TIMESERIES['ts'].data])
        with Simulation(inp_file) as sim:
            nodes = Nodes(sim)
            links = Links(sim)
            sys = SystemStats(sim)
            rg = RainGages(sim)['RG1']
            precip = 0
            cum_inflow = 0
            depth = []
            flooding = 0
            cum_flooding = 0
            cum_outfall = 0
            for st in sim:
                sim.step_advance(self.control_step)
                precip = rg.rainfall
                                
                #cum_outfall.append(nodes['Out_to_WWTP'].total_inflow)
                #cum_out_sigma = std(cum_outfall)
                #if cum_out_sigma == 0:
                #    cum_out_sigma =1
                
                depth = [nodes[tank].depth/nodes[tank].full_depth + nodes[tank].flooding*1000/nodes[tank].full_depth/self.areas[k] for k,tank in enumerate(self.tanks)]
                tank_inflow = [nodes[tank].total_inflow for k,tank in enumerate(self.tanks)]
                tank_outflow = [nodes[tank].total_outflow for k,tank in enumerate(self.tanks)]
                
                routing = sys.routing_stats
                flooding = routing['flooding'] - cum_flooding
                cum_flooding = routing['flooding']
                
                outfall = routing['outflow'] - cum_outfall
                cum_outfall = routing['outflow']

                inflow = routing['wet_weather_inflow']+routing['dry_weather_inflow']-cum_inflow
                cum_inflow = routing['wet_weather_inflow']+routing['dry_weather_inflow']
                           
                action = []
                observ = []
                for i in range(self.n_agents):
                    o = [precip/total_rain,tank_inflow[i],tank_outflow[i],depth[i]]
                    observ.append(o)
                    a = agents[i].act(o,train)
                    act = argmax(a)/(self.action_size-1)
                    links[self.orfices[i]].target_setting = act
                    action.append(a)
     
                state = [precip/total_rain,flooding/inflow,outfall/inflow]+depth
                
                observs.append(observ)      # observs (observ_size,n_agents,timesteps)
                states.append(state)    # states (state_size,timesteps)
                # if train and sim.percent_complete>0.5:
                #     continue        
                # else:
                reward = 1-flooding/inflow+outfall/inflow
                    # -abs(outfall-average(cum_outfall))/cum_out_sigma      # flooding  &  avg.outflow
                rewards.append(reward)    # rewards  (1,timesteps)
                actions.append(action)      # actions (action_size,n_agents,timesteps)
        # next_observs = observs[1:len(rewards)+1]
        # next_states = states[1:len(rewards)+1]
        # acc_rewards = [sum(array(rewards[i:])*logspace(0,len(rewards)-i-1,num=len(rewards)-i,base=self.disc)) for i,_ in enumerate(rewards)]
        next_observs = observs[1:]
        next_states = states[1:]
        # observs,states = observs[:len(rewards)],states[:len(rewards)]
        observs,states,rewards,actions = observs[:-1],states[:-1],rewards[1:],actions[:-1]
        return observs,next_observs,states,next_states,rewards,actions,cum_inflow
 

                    
    def read_rpt(self,rpt_path):
        rpt = read_rpt_file(rpt_path)
        flooding = rpt.flow_routing_continuity['Flooding Loss']['Volume_10^6 ltr']
        cum_inflow = rpt.flow_routing_continuity['Wet Weather Inflow']['Volume_10^6 ltr']+rpt.flow_routing_continuity['Dry Weather Inflow']['Volume_10^6 ltr']
        return 1-flooding/cum_inflow
        
    def test(self,f,agents,rain,test_inp_file = None):
        ts = self.inp.TIMESERIES['ts']
        ts.data = rain
        # new_inp_file = './model/arg.inp'
        self.test_inp_file = test_inp_file if test_inp_file is not None else self.test_inp_file
        self.inp.write_file(self.test_inp_file)
        observs,next_observs,states,next_states,rewards,actions,cum_inflow = self.run_simulation(agents,train=False)
        loss = f._test_loss(observs,next_observs,states,next_states,rewards,actions)
        act = [[argmax(a) for a in act]for act in actions]
        con_reward = self.read_rpt(self.test_inp_file.replace('inp','rpt'))
        print('Control Reward:   %s'%con_reward)
        return con_reward,act,loss
        
    def test_bc(self,rain,BC_inp_file = None):
        self.BC_inp_file = BC_inp_file if BC_inp_file is not None else self.BC_inp_file
        ts = self.BC_inp.TIMESERIES['ts']
        ts.data = rain
        self.BC_inp.write_file(self.BC_inp_file)
        rpt_path,_ = swmm5_run(self.BC_inp_file)
        bc_reward = self.read_rpt(rpt_path)
        print('Baseline Reward:   %s'%bc_reward)
        return bc_reward
    
    def test_efd(self,rain,EFD_inp_file = None):
        self.EFD_inp_file = EFD_inp_file if EFD_inp_file is not None else self.EFD_inp_file
        ts = self.EFD_inp.TIMESERIES['ts']
        ts.data = rain
        self.EFD_inp.write_file(self.EFD_inp_file)
        rpt_path,_ = swmm5_run(self.EFD_inp_file)
        efd_reward = self.read_rpt(rpt_path)
        print('EFD Reward:   %s'%efd_reward)
        return efd_reward