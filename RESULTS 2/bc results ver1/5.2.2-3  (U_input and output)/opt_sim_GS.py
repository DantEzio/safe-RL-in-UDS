# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:27:53 2021

@author: chong
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:55:54 2021

@author: chong
"""

import numpy as np
import get_rpt
import set_datetime
import xlrd

import get_output#从out文件读取水动力初值
import set_pump#生成下一时段的inp
import change_rain#随机生成降雨序列

import random

import datetime

from pyswmm import Simulation



'''
date_time=['08:00','08:30','09:00','09:30','10:00','10:30',\
           '11:00','11:30','12:00']
date_t=[0,30,60,90,120,150,180,210,240]

date_time=['08:00','08:20','08:40','09:00','09:20','09:40','10:00','10:20','10:40',\
           '11:00','11:20','11:40','12:00']
date_t=[0,20,40,60,80,100,120,140,160,180,200,220,240]

pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2']}
limit_level=[0.9,3.02,0.9,3.02,0.9,1.26,1.53,1.71]
pool_name=['CC-storage','CC-storage','JK-storage','JK-storage','XR-storage','XR-storage','XR-storage','XR-storage']
max_depth=[5.6,4.8,7.72]
'''

def discount_reward(r):
    gamma=0.99
    discounted_r=np.zeros_like(r)
    running_add=0
    for t in reversed(range(r.size)):
        running_add=running_add*gamma+r[t]
        discounted_r[t]=running_add
    return discounted_r

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


def read_data(st):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[0]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(nrows):
        tem=[]
        for j in range(ncols):
            tem.append(table.cell(i,j).value)     
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows,ncols)
    print(t_data.shape)
    return t_data


#########################################################################################
def initPopulation(lifeCount,geneLength):
    """初始化种群"""
    lives = []
    for i in range(lifeCount):
        #gene = [0,1,…… ,self.geneLength-1]
        #事实就是0到33
        gene=[]
        for j in range(geneLength):
            gene.append(random.randint(0,1))
            #gene.append(1)
            #将0到33序列的所有元素随机排序得到一个新的序列
            #Life这个类就是一个基因序列，初始化life的时候,两个参数，一个是序列gene，一个是这个序列的初始适应度值
            # 因为适应度值越大，越可能被选择，所以一开始种群里的所有基因都被初始化为-1
            #把生成的这个基因序列life填进种群集合里
        
        lives.append(gene)
    return lives


def cross(parent1, parent2):
    """交叉"""
    geneLength=len(parent1)
    index1 = random.randint(0, geneLength - 1)
    index2 = random.randint(index1, geneLength - 1)
    tempGene = parent2[index1:index2]                      #交叉的基因片段
    newGene = parent1
    newGene[index1:index2]=tempGene
    return newGene


def  mutation(gene):
    """突变"""
    #相当于取得0到self.geneLength - 1之间的一个数，包括0和self.geneLength - 1
    geneLength=len(gene)
    #print(geneLength)
    index1 = random.randint(0, geneLength - 1)
    index2 = random.randint(0, geneLength - 1)
    #把这两个位置的城市互换
    t=gene[index1]
    gene[index1]=gene[index2]
    gene[index2]=t
    #突变次数加1
    #mutationCount += 1
    return gene


def getOne(lives,scores,bounds):
    """选择一个个体"""
    #产生0到（适配值之和）之间的任何一个实数
    r = random.uniform(0, bounds)
    for i in range(len(lives)):
        r -= scores[i]
        if r <= 0:
            return lives[i]

    raise Exception("选择错误", bounds)


def newChild(crossRate,mutationRate,lives,scores,bounds):
    """产生新后的"""
    parent1 = getOne(lives,scores,bounds)
    rate = random.random()

    #按概率交叉
    if rate < crossRate:
        #交叉
        parent2 = getOne(lives,scores,bounds)
        gene = cross(parent1, parent2)
    else:
        gene = parent1

    #按概率突变
    rate = random.random()
    if rate < mutationRate:
        gene = mutation(gene)

    return gene


#########################################################################################
            

#rain_data=read_data('rain.xlsx')
#pumps=[]
#for pool in pool_list:
#    for item in pump_list[pool]:
#        pumps.append(item)
#
#
#A=[182,142,104,148,172,121]
#C=[0.9,0.8,0.5,0.5,0.7,0.7]
#P=[3,5,2,4,3,4]
#R=[0.2,0.3,0.4,0.5,0.6,0.7]
#b=12
#n=0.77
#crossRate=0.7
#mutationRate=0.02
#lifeCount=10
##反复多次进行模拟
#examples=[4,6]
#examples=[3]

#修改降雨数据，进行多场降雨模拟计算
def opt_GA_sim(startfile,simfile,crossRate,mutationRate,lifeCount,stepNum,date_time,pumps,raindata_ind):
    
    
    #rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
    rainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')*5#读取测试降雨数据
    #testRainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')#读取真实降雨数据
    rainnum,m=rainData.shape
    rainData=np.hstack((rainData,np.zeros((rainnum,m))))
    #testRainData=np.hstack((testRainData,np.zeros((testRainData.shape[0],m))))
    
    
    change_rain.change_rain(rainData[raindata_ind],startfile+'.inp')
    change_rain.copy_result(simfile+'.inp',startfile+'.inp')#将修改了rain数据的infile inp文件进行复制    
    
    action_seq=[]
    #用优化算法生成控制策略二维矩阵action_seq

    #初始化
    lives=initPopulation(lifeCount,len(date_time)*len(pumps))
    scores=[]
    bounds=0
    generation=0
    
    #init, first step
    for gene in lives:
        
        tem=np.array(gene)

        action_seq=list(tem.reshape(len(date_time),len(pumps)))#25*8的数组
        #print(action_seq)
        change_rain.copy_result(simfile+'.inp',startfile+'.inp')#将startfile复制为infile

        set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
        simulation(simfile+'.inp')
        #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
        #获取rpt内信息，产生新的action
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
        scores.append(1/(1+flooding))
        
        score=1/(1+flooding)
        bounds+=score
    best=lives[scores.index(max(scores))]
    #print(best)
    #初始化end
    
    log_score=[]
    for i in range(stepNum):
        if np.mod(i,10)==0:
            print('steps: ',i)
        #评估，计算每一个个体的适配值
        newLives = []
        newLives.append(best)#把最好的个体加入下一代
        while len(newLives) < lifeCount:
            newLives.append(newChild(crossRate,mutationRate,lives,scores,bounds))
        lives = newLives
        generation += 1
        
        scores=[]
        bounds=0
        #print('step'+str(i))
        for gene in lives:
            #print(action_seq)
            tem=np.array(gene)
            action_seq=list(tem.reshape(len(date_time),len(pumps)))
            
            change_rain.copy_result(simfile+'.inp',startfile+'.inp')
            set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
            simulation(simfile+'.inp')
            #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
            #获取rpt内信息，产生新的action
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
            score=1/(1+flooding)
            scores.append(score)
            
            bounds+=score
        best=lives[scores.index(max(scores))]
        max_scors=max(scores)
        log_score.append(max_scors)
        #print(i,'  ',end-begin)
        
    #最佳策略的模拟结果
    tem=np.array(best)
    action_seq=tem.reshape(len(date_time),len(pumps))
            
    change_rain.copy_result(simfile+'.inp',startfile+'.inp')
    set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
    simulation(simfile+'.inp')
    total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
    score=1/(1+flooding)

    #print('search done, time: ',end-begin)
    
    #保存训练的inp与rpt文件
    #if(trainBool==True):
    #np.savetxt('./opt_test_result/opt_logscore_'+str(raindata_ind)+'.txt', log_score, fmt="%d")
    copy_result('./opt_test_result/opt_'+str(raindata_ind)+'.inp',simfile+'.inp')
    copy_result('./opt_test_result/opt_'+str(raindata_ind)+'.rpt',simfile+'.rpt')
#   if(testBool==True):
#        copy_result('./test_result/GA/GA_'+str(iten)+'.inp',simfile+'.inp')
#        copy_result('./test_result/GA/GA_'+str(iten)+'.rpt',simfile+'.rpt')
    #print("操控序列：",action_seq.tolist())
    #print("得分：",reward_sum)
    
    #np.savetxt('./sim/GAActionSeq.txt',action_seq,fmt='%f',delimiter=',')
    return action_seq

if __name__=='__main__':
    startfile='./sim/orf'
    simfile='./sim/opt_using'
    crossRate,mutationRate,lifeCount,stepNum=0.7,0.03,20,100
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00']
    pumps=['CC-Pump-1','CC-Pump-2','JK-Pump-1','JK-Pump-2','XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']
    
    for raindata_ind in range(4):
        print('rain num: ',raindata_ind)
        aq=opt_GA_sim(startfile,simfile,crossRate,mutationRate,lifeCount,stepNum,date_time,pumps,raindata_ind)
        np.savetxt('./opt_test_result/opt_'+str(raindata_ind)+'.txt', aq, fmt="%d")
    
    