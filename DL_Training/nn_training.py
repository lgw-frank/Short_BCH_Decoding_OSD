# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
from tensorflow.keras import  metrics
import nn_net as CRNN_DEF
from collections import Counter
import  os
from typing import Any, Dict,Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import pickle

def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            # If model.inputs is invalid, try to use the input to get concrete
            # function for model.call (subclass model).
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            print('Failed to count model FLOPs with error %s, because the build() '
                 'methods in keras layers were not called. This is probably because '
                 'the model was not feed any input, e.g., the max train step already '
                 'reached before this run.', e)
            return None
    return None
def print_flops(model):
    flops = try_count_flops(model)
    print(flops/1e3,"K Flops")
    return None
def print_model_summary(model):
    # Create an instance of the model
    #model = ResNet(num_blocks=3, filters=64, kernel_size=3)
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Print model summary
    list_length = GL.get_map('num_iterations')+1
    stripe = GL.get_map('stripe')
    model.build(input_shape=(None, list_length,stripe))  # Assuming input shape is (batch_size, sequence_length, input_dim)
    model.summary()
    return None
def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) 
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return start_step,ckpt_f

def calculate_loss(original_input_list,refined_inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    list_length = GL.get_map('num_iterations')+1
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch    
    if not len(original_input_list):
        loss_list = []*list_length
    else:
        loss_list = []
        for i in range(list_length):     
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-original_input_list[i], labels=labels))
            loss_list.append(loss)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-refined_inputs, labels=labels))
    loss_list.append(loss)
    #tf.print(loss_list)
    return  loss_list

def evaluate_MRB_bit(updated_inputs,labels):
    inputs_abs = tf.abs(updated_inputs)
    code = GL.get_map('code_parameters')
    order_index = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(updated_inputs,order_index,batch_dims=1)
    order_inputs_hard = tf.where(order_inputs>0,0,1)
    order_labels = tf.cast(tf.gather(labels,order_index,batch_dims=1),tf.int32)
    cmp_result = tf.reduce_sum(tf.where(order_inputs_hard[:,-code.k:] == order_labels[:,-code.k:],0,1),axis=-1).numpy()
    Demo_result=Counter(cmp_result) 
    #print(Demo_result)
    return Demo_result

def dic_union(dicA,dicB):
    for key,value in dicB.items():
        if key in dicA:
            dicA[key] += value
        else:
            dicA[key] = value
    return dict(sorted(dicA.items(), key=lambda d:d[0])) 



def Training_NN(selected_ds,restore_info,indicator_list,prefix_list,DIA=False):
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    list_length = GL.get_map('num_iterations')+1
    decoder_type = GL.get_map('selected_decoder_type')
    snr_info = str(snr_lo)+'-'+str(snr_hi)
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    prefix = 'benchmark'     
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list) 
    if DIA:
        epochs = GL.get_map('epochs') 
        cnn = CRNN_DEF.conv_bitwise()
        rnn1 = CRNN_DEF.rnn_one()
        rnn2 = CRNN_DEF.rnn_two()
        nn_list = [cnn,rnn1,rnn2]    
        for i,element in enumerate(indicator_list):
            if element == True:
                nn = nn_list[i]
                prefix = prefix_list[i]
                break
        step = 0
        exponential_decay = GL.optimizer_setting()
        optimizer = tf.keras.optimizers.legacy.Adam(exponential_decay)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=nn, myAwesomeOptimizer=optimizer)
        summary_writer,manager_current = GL.log_setting(restore_info,checkpoint,prefix)
        #unpack related info for restoraging
        [ckpts_dir,ckpt_nm,restore_step] = restore_info  
        if restore_step:
            print('Load the previous saved model from disk!')
            step,ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
            status = checkpoint.restore(ckpt_f)
            status.expect_partial()  
        loss_meter = metrics.Mean()
        #initialize starting point
        start_epoch = step//num_counter
        residual = step%num_counter
        jump_loop = False
        if GL.get_map('nn_train'):  
            if step < GL.get_map('termination_step'):
                for epoch in range(start_epoch,epochs):
                    print("\nStart of epoch %d:" % epoch) 
                    for i in range(residual,num_counter):
                        squashed_inputs,super_input_matrix,labels = nn.preprocessing_inputs(input_list[i])
                        step = step + 1
                        with tf.GradientTape() as tape:
                            refined_inputs = nn(squashed_inputs)
                            loss_list = calculate_loss(super_input_matrix,refined_inputs,labels)
                            loss = loss_list[-1]
                            loss_meter.update_state(loss) 
                        total_variables = nn.trainable_variables         
                        grads = tape.gradient(loss,total_variables)
                        grads_and_vars=zip(grads, total_variables)
                        capped_gradients = [(tf.clip_by_norm(grad,5), var) for grad, var in grads_and_vars if grad is not None]
                        optimizer.apply_gradients(capped_gradients)   
                        if step % print_interval == 0:   
                            print('Step:%d  Loss:%.3f'%(step,loss.numpy()))
                            #_ = evaluate_MRB_bit(inputs,labels)
                            #_ = evaluate_MRB_bit(refined_inputs,labels)                                                               
                        if step % record_interval == 0:
                            manager_current.save(checkpoint_number=step)                   
                        if step >= GL.get_map('termination_step'):
                            jump_loop = True
                            break
                        loss_meter.reset_states()  
                    residual = 0
                    if jump_loop:
                        break
                #save the latest setting
                manager_current.save(checkpoint_number=step) 

    #verifying trained pars from start to end    
    dic_sum = {}
    dic_sum_initial = {}
    dic_sum_end = {}
    actual_size = 0
    loss_sum = [0.]*(list_length+1)
    #query partition of MRB
    for i in range(num_counter):      
        if DIA:
            squashed_inputs,super_input_matrix,labels = nn.preprocessing_inputs(input_list[i])
            updated_inputs = nn(squashed_inputs)
            #nn.print_model()        
        else:
            labels = input_list[i][1][0::list_length]
            updated_inputs = input_list[i][0][0::list_length]
            super_input_matrix = []
        loss_list = calculate_loss(super_input_matrix,updated_inputs,labels)
        # Element-wise addition
        loss_sum = [x + y for x, y in zip(loss_sum, loss_list)]
        
        actual_size += updated_inputs.shape[0]

        cmp_results = evaluate_MRB_bit(updated_inputs,labels)
        dic_sum = dic_union(dic_sum,cmp_results)
        cmp_results = evaluate_MRB_bit(input_list[i][0][0::list_length],labels)
        dic_sum_initial = dic_union(dic_sum_initial,cmp_results)
        cmp_results = evaluate_MRB_bit(input_list[i][0][list_length-1::list_length],labels)
        dic_sum_end = dic_union(dic_sum_end,cmp_results)
        #query pattern distribution
        if (i+1)%record_interval == 0:
            print(dic_sum) 

    average_loss = [loss_sum[i]/actual_size for i in range(list_length+1)]
    print(f'Total counts:{actual_size}')
    formatted_loss = [f'{value:.4f}' for value in average_loss]
    print("\nAverage Loss:")
    print('\t'.join(formatted_loss))
    print('Summary for 0-th dist:',dic_sum_initial)
    print('Summary for T-th dist:',dic_sum_end)
    print('Summary before GE for '+prefix+':',dic_sum)
    #save on disk files
    log_dir = './log/'+decoder_type+'/'+snr_info+'dB/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
    with open(log_dir+"dist-error-pattern-"+prefix+".pkl", "wb") as fh:
        pickle.dump(actual_size,fh)
        pickle.dump(dic_sum_initial,fh)        
        pickle.dump(dic_sum_end,fh)
        pickle.dump(dic_sum,fh)
