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

def get_rpt_flooding(filename):
    with open(filename, 'rt') as data:
        flooding=0
        pumps_flag =  False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('Flooding Loss')>=0:
                    flooding=float(node[4])
                    pumps_flag=False

    return flooding
        

if __name__ == '__main__':

    filename='./final-noAFI-drain/HC/0.rpt'
    #arg_output_path0 = './sim/arg-original.rpt'
    print(get_rpt_flooding(filename))

