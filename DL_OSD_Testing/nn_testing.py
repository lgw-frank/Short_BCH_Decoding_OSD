# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
import nn_net as CRNN_DEF
import ordered_statistics_decoding as OSD_mod
import os
import numpy as  np
import time
from itertools import combinations

# import numpy as np
# import ms_decoder_dense as MDL
#import ordered_statistics_decoding as OSD_Module
#import ordered_statistics_decoding as OSD_Module

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return ckpt_f    

def NN_gen(restore_info,indicator_list):  
    cnn = CRNN_DEF.conv_bitwise()
    rnn1 = CRNN_DEF.rnn_one()
    rnn2 = CRNN_DEF.rnn_two()
    nn_list = [cnn,rnn1,rnn2] 
    for i,element in enumerate(indicator_list):
        if element == True:
            nn = nn_list[i]
            break
    checkpoint = tf.train.Checkpoint(myAwesomeModel=nn)
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,restore_step] = restore_info 
    if restore_step:
        ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial()       
    return nn

def query_nn_type(indicator_list,prefix_list,DIA):
    nn_type = 'benchmark'
    if DIA: 
        for i,element in enumerate(indicator_list):
            if element == True:
                nn_type = prefix_list[i]
                break
    return nn_type

def binary_sequences_matrix(n, p):
    sequences = []
    for weight in range(p + 1):  # Hamming weight from 0 to p
        for indices in combinations(range(n), weight):
            sequence = [0] * n
            for idx in indices:
                sequence[idx] = 1
            sequences.append(sequence)
    return np.array(sequences)  # Convert to a NumPy matrix for cleaner output   
    
def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss

def calculate_list_cross_entropy_ber(input_list,labels):
    cross_entropy_list = []
    ber_list = []
    for i in range(len(input_list)):
        cross_entropy_element = calculate_loss(input_list[i],labels).numpy()
        cross_entropy_list.append(cross_entropy_element)
        current_hard_decision = tf.where(input_list[i]>0,0,1)
        compare_result = tf.where(current_hard_decision!=labels,1,0)
        num_errors = tf.reduce_sum(compare_result)
        ber_list.append(num_errors)
    return cross_entropy_list,ber_list
     

def Testing_OSD(snr,selected_ds,restore_info,indicator_list,prefix_list,DIA):
    start_time = time.process_time()
    code = GL.get_map('code_parameters')
    order = GL.get_map('threshold_order')
    osd_instance = OSD_mod.osd(code)
    nn_type = query_nn_type(indicator_list,prefix_list,DIA)
    teps_matrix = binary_sequences_matrix(code.k,order)
    #acquire decoding 
    nn = NN_gen(restore_info,indicator_list) 
    logdir = './log/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = logdir+'OSD-'+str(order)+'-'+nn_type+'.txt'  
    list_length = GL.get_map('num_iterations')+1
    
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list)      
    
    fail_sum = 0
    correct_sum = 0
    windows_sum = 0
    complexity_sum = 0
    actual_size = 0
    cross_entropy_list_sum = [0.]*(list_length+1)
    ber_list_sum = [0]*(list_length+1)
    
    for i in range(num_counter):   
        if DIA:
            squashed_inputs,_,labels = nn.preprocessing_inputs(input_list[i])
            new_inputs = nn(squashed_inputs)
            #nn.print_model()
        else:
            labels = input_list[i][1][0::list_length]
            new_inputs = input_list[i][0][0::list_length]
        actual_size += labels.shape[0]
        
        input_data_list = [input_list[i][0][j::list_length] for j in range(list_length)]
        input_data_list.append(new_inputs)
        cross_entropy_list,ber_list = calculate_list_cross_entropy_ber(input_data_list,labels)
        # Element-wise addition using a loop
        cross_entropy_list_sum = [a + b for a, b in zip(cross_entropy_list, cross_entropy_list_sum)] 
        ber_list_sum = [a + b for a, b in zip(ber_list, ber_list_sum)]
        # Alternatively, using map and lambda function
        # result = list(map(lambda x, y: x + y, list1, list2))
        #OSD processing
        order_list = osd_instance.execute_osd(input_list[i][0],new_inputs,labels,teps_matrix)
        correct_counter,fail_counter = osd_instance.best_estimate(order_list)
        correct_sum += correct_counter
        fail_sum += fail_counter  
        if (i+1)%10 == 0:
            average_size = round(complexity_sum/actual_size,4)
            wins_size = round(windows_sum/actual_size,4)
            print(f'\nFor {snr:.1f}dB order:{order}:')
            print(f'--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} Wins:{wins_size}')      
            average_loss_list  = [cross_entropy_list_sum[j]/actual_size for j in range(list_length+1)] 
            average_ber_list  = [ber_list_sum[j]/(actual_size*code.check_matrix_col) for j in range(list_length+1)]
            formatted_floats_ce = [" ".join(["{:.3f}".format(value) for value in average_loss_list])]
            formatted_floats_ber = [" ".join(["{:.3f}".format(value) for value in average_ber_list])]
            print(f'avr CE per itr:\n{formatted_floats_ce} \nBER:{formatted_floats_ber}')
            T2 =time.process_time()
            print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
        if i == num_counter-1 or fail_sum >= GL.get_map('termination_threshold'):
            break
    T2 =time.process_time()
    FER = round(fail_sum/actual_size,5)  
    average_size = round(complexity_sum/actual_size,4)
    wins_size = round(windows_sum/actual_size,4)
    average_loss_list  = [cross_entropy_list_sum[j]/actual_size for j in range(list_length+1)]  
    average_ber_list  = [ber_list_sum[j]/(actual_size*code.check_matrix_col) for j in range(list_length+1)]  
    print('\nFor %.1fdB (order %d of size %d) '%(snr,order,teps_matrix.shape[0])+nn_type+':\n')
    print('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')          
    print(f'FER:{FER}--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} Wins:{wins_size}')
    formatted_floats_ce = [" ".join(["{:.3f}".format(value) for value in average_loss_list])]
    formatted_floats_ber = [" ".join(["{:.3f}".format(value) for value in average_ber_list])]
    print(f'avr CE per itr:\n{formatted_floats_ce} \nBER:{formatted_floats_ber}')
    print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
    with open(log_filename,'a+') as f:
        f.write(f'For {snr:.1f}dB order {order} of size {teps_matrix.shape[0]}:\n')
        f.write('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')         
        f.write(f'FER:{FER}--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} Wins:{wins_size}\n')
        formatted_floats_ce = [" ".join(["{:.3f}".format(value) for value in average_loss_list])]
        formatted_floats_ber = [" ".join(["{:.3f}".format(value) for value in average_ber_list])]
        f.write(f'avr CE per itr:\n{formatted_floats_ce} \nBER:{formatted_floats_ber}\n')
        f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!\n')
    return FER,log_filename

  
