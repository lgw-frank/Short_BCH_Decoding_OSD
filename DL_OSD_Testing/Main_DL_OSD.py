# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:09:34 2023

@author: zidonghua_30
"""
import time
T1 = time.process_time()
import sys
import globalmap as GL
import nn_testing as NN_test

sys.argv = "python 0.0 3.5 8 100 4 BCH_63_45_3_strip.alist NMS-1".split()

#setting a batch of global parameters

DIA = True

GL.global_setting(sys.argv) 
selected_ds,snr_list = GL.data_setting()
restore_info = []
indicator_list = []
prefix_list = []

# one and only one of following will be true
CNN_indicator = True
RNN1_indicator = False
RNN2_indicator = False
indicator_list = [CNN_indicator,RNN1_indicator,RNN2_indicator]
prefix_list = ['model_cnn','model_rnn1','model_rnn2']

restore_info = GL.logistic_setting_model(indicator_list,prefix_list)
    
FER_list =  []
for i in range(len(snr_list)):
  snr = round(snr_list[i],2)
  FER,log_filename = NN_test.Testing_OSD(snr,selected_ds[i],restore_info,indicator_list,prefix_list,DIA)
  FER_list.append((snr,FER))
print(f'Summary of FER:{FER_list}')
with open(log_filename,'a+') as f:
  f.write(f'\n Summary of FER:{FER_list}')
T2 =time.process_time()
print('Running time:%s seconds!'%(T2 - T1))