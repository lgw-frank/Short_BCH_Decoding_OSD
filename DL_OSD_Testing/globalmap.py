"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix
import os
import numpy as np

map = {}
def set_map(key, value):
    map[key] = value
def del_map(key):
    try:
        del map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")
def get_map(key):
    try:
        if key in "all":
            return map
        return map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")

#global parameters setting
def global_setting(argv):
    #command line arguments
    set_map('snr_lo', float(argv[1]))
    set_map('snr_hi', float(argv[2]))
    set_map('snr_num',int(argv[3]))
    set_map('unit_batch_size',int(argv[4]))
    set_map('num_iterations', int(argv[5]))
    set_map('H_filename', argv[6])
    set_map('selected_decoder_type',argv[7])
    
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)      
    #filling parity check matrix info
    set_map('regular_matrix',True)
    set_map('generate_extended_parity_check_matrix',False)
    set_map('reduction_iteration',3)    
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    
    set_map('termination_threshold',500)
    set_map('threshold_order',2)     #threshold for number of non-zero elements across MRB
    set_map('training_snr',2.6)

    #store it onto global space
    set_map('code_parameters', code)
    set_map('print_interval',100)
    set_map('record_interval',100) 
 
def logistic_setting_model(indicator_list,prefix_list):
    for i,element in enumerate(indicator_list):
        if element == True:
            prefix = prefix_list[i]
            break
    n_iteration = get_map('num_iterations')
    training_snr = get_map('training_snr')
    snr_lo = training_snr
    snr_hi = training_snr
    snr_info = '/'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    ckpts_dir = '../DL_Training/ckpts/'+prefix+snr_info+str(n_iteration)+'th'+'/'
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)   
    ckpt_nm = 'bch-ckpt'  
    restore_model_step = 'latest'
    restore_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_model_info 

def data_setting():
    code = get_map('code_parameters')
    n_dims = code.check_matrix_col
    batch_size = get_map('unit_batch_size')
    snr_num = get_map('snr_num')
    snr_lo = get_map('snr_lo')
    snr_hi = get_map('snr_hi')
    snr_list = np.linspace(snr_lo,snr_hi,snr_num)
    n_iteration = get_map('num_iterations')
    list_length = n_iteration+1
    data_handler_list = []
    data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    decoder_type = get_map('selected_decoder_type')
    if get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        file_name = 'bch-allzero-retest.tfrecord'
    else:
        file_name = 'bch-nonzero-retest.tfrecord'    
    for i in range(snr_num):
        snr = str(round(snr_list[i],2))
        input_dir = data_dir+decoder_type+'/'+str(n_iteration)+'th/'+snr+'dB/'
        # reading in training/validating data;make dataset iterator
        file_dir = input_dir+file_name
        dataset_test = Reading.data_handler(code.check_matrix_col,file_dir,batch_size*list_length)
        data_handler_list.append(dataset_test)
        
    return data_handler_list,snr_list               
                   