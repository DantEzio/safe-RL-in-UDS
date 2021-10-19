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
        pumps_flag = pumps_flag_end = outfall_flag_end = outfall_flag= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            pumps_flag_end = handle_line(line, pumps_flag_end, 'Quality Routing Continuity')
            outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            outfall_flag_end=handle_line(line, outfall_flag_end, 'Link Flow Summary')
            
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
                if line.find('outfall-27')>=0 or line.find('outfall-28')>=0 or line.find('outfall-24')>=0:
                    upflow+=float(node[5])
                elif line.find('outfall-')>=0 or line.find('12')>=0 or line.find('67')>=0:
                    downflow+=float(node[5])
            
            if outfall_flag_end:
                outfall_flag=False

    return total_in,flooding,store,outflow,upflow,downflow


def get_safety1(filename):
    with open(filename, 'rt') as data:
        N_wl=0
        
        nodeD_flag =  False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            nodeD_flag = handle_line(line, nodeD_flag, 'Node Depth Summary')
            node = line.split() # Split the line by whitespace      
            if nodeD_flag and node!=[]:
                if line.find('WS02006235')>=0:
                    N_wl=float(node[2])
                    nodeD_flag=False
    return N_wl

def get_safety2(filename):
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

def get_safety3(filename):
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
                if line.find('WS')>=0 or line.find('YS')>=0 or \
                   line.find('JK')>=0 or line.find('XK')>=0 or \
                   line.find('CC')>=0:
                    sev+=float(node[5])*float(node[1])
                    
            if total_flag and node!=[]:
                if line.find('Dry Weather Inflow')>=0 or line.find('Wet Weather Inflow')>=0:
                    inflow+=float(node[5])
    return sev/inflow
        

if __name__ == '__main__':

    filename='./sim/staf.rpt'
    #arg_output_path0 = './sim/arg-original.rpt'
    #print(get_rpt(filename))
    print(get_safety1(filename))
    print(get_safety2(filename))
    print(get_safety3(filename))