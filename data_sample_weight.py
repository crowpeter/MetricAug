#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:48:07 2022

@author: crowpeter
"""
import random
import numpy as np
#%% algorithm tesing
def sampling_w_compute(total_weight, mini_remain, va_glob_wf1):
    clean_wf1 = va_glob_wf1[-1]
    va_wf1s = [va_glob_wf1[i] for i in range(5)]
    
    M = 1
    gap =  np.array([clean_wf1 - va_wf1 for va_wf1 in va_wf1s])
    metric_weight = np.array([gap[i]/abs(sum(gap)) for i in range(5)])
    less_than_lower_bound_num = len(np.where(metric_weight <= mini_remain)[0])

    while True:
        # initialize
        total_weight = M
        # assign: metric < lower_bound = lower_bound
        metric_weight[np.where(metric_weight <= mini_remain)] = mini_remain
        
        # re-scale total weight
        total_weight -= len(np.where(metric_weight <= mini_remain)[0] == mini_remain) * mini_remain
        
        # assign: metric weight
        for i in np.where(metric_weight > mini_remain)[0]:
        # for i in np.where(gap > mini_remain)[0]:
            metric_weight[i] = (gap[i]/gap[np.where(metric_weight > mini_remain)].sum())*total_weight
            
        # re_compute lower_bound number in metric_weight
        less_than_lower_bound_num = len(np.where(metric_weight < mini_remain)[0])
    
        if round(sum(metric_weight),2) == M and less_than_lower_bound_num == 0:
            break
        
    metric_weight = {i:metric_weight[i] for i in range(len(metric_weight))}
    return metric_weight

if __name__ == '__main__':
    TOTAL_W = 1
    MIN_RM = 0.05
    RE_BOX = {}
    metric_init_weight = {0:0.2, 1:0.2, 2:0.2, 3:0.2, 4:0.2}
    metric_weight = metric_init_weight.copy()
    
    # #%%
    clean_wf1 = 0.6
    va_wf1s = np.array([random.uniform(0, 0.5) for i in range(0,5)])
    va_wf1s_dict = {-1:clean_wf1}
    for i in range(5):
        va_wf1s_dict[i] = va_wf1s[i]
    
    metric_w = sampling_w_compute(TOTAL_W, MIN_RM, va_wf1s_dict)
    print('New data aug weight:', metric_w)
    print(round(sum([metric_weight[key] for key in metric_w.keys()]),2))