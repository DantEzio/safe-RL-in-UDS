# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:47:58 2018

@author: chong
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:23:08 2018

@author: chong
"""

#from yaml import load
import math
import matplotlib.pyplot as plt

#constants = load(open('./constants.yml', 'r', encoding='utf-8'))


def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def replace_line(line, title,rain,t,rg):

    node=line.split()
    if(node[0]==rg+'Oneyear-2h'):
        t=t+1
        tem=node[0]+' '*8+node[1]+' '+node[2]+' '*6+str(rain)
        #print(tem)
        line=tem
        return t,line
    else:
        return t,line


def handle_line(line, flag, title,rain,t,rg):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    elif line.find(';') == -1 and flag:
        t,line = replace_line(line, title,rain,t,rg)
    return t,line, flag


def change_rain(rain,infile,rg):
    temfile=infile+'tem_rain.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        rain_flag =  False
        t=0
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            t,line, flag = handle_line(line, rain_flag, '[TIMESERIES]',rain[t],t,rg)
            #rg用于指明rain替换哪个位置
            rain_flag = flag
            output.write(line + '\n')
    output.close()
    copy_result(infile,temfile)


def gen_rain(t,A,C,P,b,n,R,deltt):
    '''
    t是生成雨量时间序列步数上限
    delt是时间间隔，取1
    '''
    rain=[]
    for i in range(t):
        if i <int(t*R):
            rain.append(A*(1+C*math.log(P))/math.pow(((t*R-i)+b),n))
        else:
            rain.append(A*(1+C*math.log(P))/math.pow(((i-t*R)+b),n))
    
    return rain

if __name__ == '__main__':
    infile='orf.inp'
    outfile='tem.inp'
    A=10#random.randint(5,15)
    C=13#random.randint(5,20)
    P=2#random.randint(1,5)
    b=1#random.randint(1,3)
    n=0.5#random.random()
    R=0.5#random.random()
    deltt=1
    t=288
    #change_rain(A,C,P,b,n,infile,outfile)
    rain1=gen_rain(t,A,C,P,b,n,R,deltt)
    rain2=gen_rain(t,A,C,P,b,n,R,deltt)
    rain3=gen_rain(t,A,C,P,b,n,R,deltt)
    rain4=gen_rain(t,A,C,P,b,n,R,deltt)
    plt.plot(range(288),rain1)
    #copy_result(infile,'arg-original.inp')
    change_rain(rain1,infile,'RG1')
    change_rain(rain2,infile,'RG2')
    change_rain(rain3,infile,'RG3')
    change_rain(rain4,infile,'RG4')