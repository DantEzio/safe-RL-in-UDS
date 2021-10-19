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

#constants = load(open('./constants.yml', 'r', encoding='utf-8'))
from swmm_api import read_rpt_file

def handle_line(line, flag, title):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    return flag

def get_rpt(filename):
    with open(filename, 'rt') as data:
        total_in=0
        flooding=0
        store=0
        outflow=0
        upflow=0
        downflow=0
        pumps_flag = pumps_flag_end = outfall_flag_end = outfall_flag= upflow_flag=upflow_flag_end= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            pumps_flag_end = handle_line(line, pumps_flag_end, 'Highest Continuity Errors')
            outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            outfall_flag_end=handle_line(line, outfall_flag_end, 'Link Flow Summary')
            upflow_flag = handle_line(line, upflow_flag, 'Storage Volume Summary')
            upflow_flag_end=handle_line(line, upflow_flag_end, 'Outfall Loading Summary')
            
            
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('External Outflow')>=0 or\
                   line.find('Exfiltration Loss')>=0 or \
                   line.find('Mass Reacted')>=0:
                    outflow+=float(node[4])

                elif line.find('Flooding Loss')>=0:
                    flooding=float(node[4])
                elif line.find('Final Stored Volume')>=0:
                    store=float(node[5])
                elif line.find('Dry Weather Inflow')>=0 or \
                     line.find('Wet Weather Inflow')>=0 :
                    total_in+=float(node[5])
                    
                elif line.find('Groundwater Inflow')>=0 or line.find('RDII Inflow')>=0 or line.find('External Inflow')>=0:
                    total_in+float(node[4])
                elif line.find('Quality Routing Continuity')>=0:
                    pumps_flag=False
            
            if pumps_flag_end:
                pumps_flag=False
                    
            if outfall_flag and node!=[]:
                if line.find('Out_to_WWTP')>=0:
                    downflow+=float(node[4])
            
            if outfall_flag_end:
                outfall_flag=False
                
            if upflow_flag and node!=[]:
                if line.find('T1')>=0 or line.find('T2')>=0 or line.find('T3')>=0 or \
                   line.find('T4')>=0 or line.find('T5')>=0 or line.find('T6')>=0:
                    upflow+=float(node[1])
            
            if upflow_flag_end:
                upflow_flag=False

    return total_in,flooding,store,outflow,upflow,downflow


def get_safety1(filename):
    with open(filename, 'rt') as data:
        N_wl=[]
        
        nodeD_flag = nodeD_flag_end = False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            nodeD_flag = handle_line(line, nodeD_flag, 'Node Depth Summary')
            nodeD_flag_end = handle_line(line, nodeD_flag_end, 'Node Inflow Summary')
            node = line.split() # Split the line by whitespace      
            if nodeD_flag and node!=[]:
                if line.find('T1')>=0 or \
                   line.find('T2')>=0 or \
                   line.find('T3')>=0 or \
                   line.find('T4')>=0 or \
                   line.find('T5')>=0 or \
                   line.find('T6')>=0:
                    N_wl.append(float(node[2]))
            if nodeD_flag_end:
                nodeD_flag=False
    return N_wl

def get_safety2(filename):
    #从action中统计
    pass
'''
    with open(filename, 'rt') as data:
        num=[]
        iten=0
        pumps_flag =  False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Pumping Summary')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('CC-Pump-1')>=0 or line.find('CC-Pump-2')>=0 or \
                   line.find('JK-Pump-1')>=0 or line.find('JK-Pump-2')>=0 or \
                   line.find('XR-Pump-1')>=0 or line.find('XR-Pump-2')>=0 or \
                   line.find('XR-Pump-3')>=0 or line.find('XR-Pump-4')>=0:
                    num.append(float(node[2]))
                    iten+=1
                if iten==8:#8台泵
                    pumps_flag=False
    return num
'''

def get_safety3(filename):
    rpt = read_rpt_file(filename)
    data = rpt.node_flooding_summary.values
    sev=0
    for i in range(data.shape[0]):
        sev+=data[i,0]*data[i,3]
        
    data2=rpt.flow_routing_continuity
    inflow=data2['Dry Weather Inflow']['Volume_10^6 ltr']+data2['Wet Weather Inflow']['Volume_10^6 ltr']
    
    '''
    with open(filename, 'rt') as data:
        sev=0
        inflow=0
        pumps_flag_end=pumps_flag = total_flag= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Node Flooding Summary')
            pumps_flag_end = handle_line(line, pumps_flag_end, 'Storage Volume Summary')
            total_flag=handle_line(line, total_flag, 'Flow Routing Continuity')
            node = line.split() # Split the line by whitespace
            if pumps_flag_end:
                pumps_flag=False
            
            if pumps_flag and node!=[]:
                if line.find('T1')>=0 or line.find('T2')>=0 or \
                   line.find('T3')>=0 or line.find('T4')>=0 or \
                   line.find('T5')>=0:
                    sev+=float(node[5])*float(node[1])
                    
            if total_flag and node!=[]:
                if line.find('Dry Weather Inflow')>=0 or line.find('Wet Weather Inflow')>=0:
                    inflow+=float(node[5])
    '''
    return sev/inflow
        

if __name__ == '__main__':

    filename='./sim/staf.rpt'
    #arg_output_path0 = './sim/arg-original.rpt'
    #print(get_rpt(filename))
    #print(get_safety1(filename))
    #print(get_safety2(filename))
    print(get_safety3(filename))