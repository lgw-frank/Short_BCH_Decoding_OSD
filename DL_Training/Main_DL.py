# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:09:34 2023

@author: zidonghua_30
"""
import sys
import globalmap as GL
import nn_training as CNN_RNN
import os

# Set KMP_DUPLICATE_LIB_OK environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.argv = "python 2.6 2.6 10 4 BCH_63_45_3_strip.alist NMS-1".split()

#setting a batch of global parameters
GL.global_setting(sys.argv) 
selected_ds = GL.data_setting()

DIA = True

restore_info = []
indicator_list = []
prefix_list = []

if DIA:
    # one and only one of following will be true
    CNN_indicator = True
    RNN1_indicator = False
    RNN2_indicator = False
    indicator_list = [CNN_indicator,RNN1_indicator,RNN2_indicator]
    prefix_list = ['model_cnn','model_rnn1','model_rnn2']
    restore_info = GL.logistic_setting_model(indicator_list,prefix_list)
    
CNN_RNN.Training_NN(selected_ds,restore_info,indicator_list,prefix_list,DIA) 