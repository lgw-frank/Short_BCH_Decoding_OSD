# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
import data_generating as Data_gen
import math,random
#from distfit import distfit    
    
class Decoding_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = Decoder_Layer()
    def build(self, input_shape):
        # Ensure the layer is built with the appropriate input shape
        pass
    def call(self, inputs):
        decoding_info = self.layer(inputs)
        return decoding_info   
    def collect_failed_input_output(self,soft_output_list,labels,index):
        list_length = self.layer.num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels     
 
    def get_eval(self,soft_output_list, labels):
        soft_output = soft_output_list[-1]
        tmp = tf.cast((soft_output < 0),tf.bool)
        label_bool = tf.cast(labels, tf.bool)
        err_batch = tf.math.logical_xor(tmp, label_bool)
        FER_data = tf.reduce_any(err_batch,1)
        index = tf.where(FER_data)
        BER_data = tf.reduce_sum(tf.cast(err_batch, tf.int64))
        FER = tf.math.count_nonzero(FER_data)/soft_output.shape[0]
        BER = BER_data/(soft_output.shape[0]*soft_output.shape[1])
        return FER, BER,index    

 
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self,initial_value = 0.421):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.H = self.code.H
        self.initials = initial_value
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape):   
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
            self.shared_check_weight = self.add_weight(name='decoder_check_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))       
    # Code for model call (handles inputs and returns outputs)
    def call(self,inputs):
        soft_input = inputs[0]
        labels = inputs[1]    
        soft_output_list,loss,label,_ = self.belief_propagation_op(soft_input,labels)
        return soft_output_list,loss,label
    def get_evenly_shifted_integers(self,start,end,count):
      step = (end - start) // count
      base_integers = [start + i * step for i in range(count)]
      # Calculate the maximum possible shift to keep numbers within the range
      max_shift = end-1-base_integers[-1]
      shift = random.randint(0, max_shift)  
      shifted_integers = [x + shift for x in base_integers] 
      return shifted_integers
    def aggregate_cyclic_words(self,soft_input):
        #cycling the input from start to end
        soft_input_split = tf.concat([soft_input[:,0::2],soft_input[:,1::2]],axis=1)
        permutated_codewords = self.frobenius_automorphism(soft_input,soft_input.shape[1])
        soft_input = tf.concat([soft_input,soft_input_split,permutated_codewords],0)
        num_shifts = GL.get_map('num_shifts')
        shifted_integers = self.get_evenly_shifted_integers(0,soft_input.shape[1],num_shifts)
        shifted_input_list = [tf.roll(soft_input,shifted_integers[i],axis=1)  for i in range(num_shifts)]
        super_inputs = tf.concat(shifted_input_list,axis=0)
        return super_inputs,shifted_integers      
      
                
# builds a belief propagation TF graph

    def belief_propagation_op(self,soft_input,labels):
        soft_output_list = [soft_input]*(self.num_iterations+1)
        return tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_output_list, # soft input for this iteration
                0.,# loss
                labels,
                0, # iteration number
            ]
            )
            
    # compute messages from variable nodes to check nodes
    def compute_vc(self,soft_input):
        check_matrix_H = tf.cast(self.H,tf.float32)                                      
        vc_matrix = tf.expand_dims(soft_input,axis=1)*check_matrix_H
        return vc_matrix  
 
    # compute messages from check nodes to variable nodes
    def compute_cv(self,vc_matrix):
        check_matrix_H = self.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp1 = tf.reduce_prod(vc_matrix_sign,2)
        temp1 = tf.expand_dims(temp1,2)
        transition_sign_matrix = temp1*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,-1e30-1,0.)
        back_matrix = tf.expand_dims(back_matrix,0)
        vc_matrix_abs = tf.abs(vc_matrix)
        vc_matrix_abs_clip = tf.clip_by_value(vc_matrix_abs, 0, 1e30)
        vc_matrix_abs_minus = -tf.abs(vc_matrix_abs_clip)
        vc_decision_matrix = vc_matrix_abs_minus+back_matrix
        min_submin_info = tf.nn.top_k(vc_decision_matrix,k=2)
        min_column_matrix = -min_submin_info[0][:,:,0]
        min_column_matrix = tf.expand_dims(min_column_matrix,2)
        min_column_matrix = min_column_matrix*check_matrix_H
        second_column_matrix = -min_submin_info[0][:,:,1]
        second_column_matrix = tf.expand_dims(second_column_matrix,2)
        second_column_matrix = second_column_matrix*check_matrix_H  
        result_matrix = tf.where(vc_matrix_abs_clip>min_column_matrix,min_column_matrix,second_column_matrix)  
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
            normalized_tensor = tf.nn.softplus(self.shared_check_weight) 
        cv_matrix = normalized_tensor*result_matrix*tf.stop_gradient(result_sign_matrix)         
        return cv_matrix
    
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        penalty_weight = GL.get_map('penalty_weight')
        labels = tf.cast(labels,tf.float32)
        weighte_matrix = tf.where(tf.sign(soft_output) != 1-2*labels,float(penalty_weight),1.)
        CE_loss = tf.reduce_sum(weighte_matrix*tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss
    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input,shifted_integers,labels,iteration,soft_output_list):
        #tmp_labels = self.interleave_columns(labels[:,::2],labels[:,1::2]) 
        # print(np.sum(self.code.H@tf.transpose(tmp_labels)%2))
        # print(np.sum(self.code.H@tf.transpose(labels)%2))
        temp = tf.reduce_sum(cv_matrix,1)
        #aligning with cycled input
        num_shifts = GL.get_map('num_shifts')
        batch_size = soft_input.shape[0]//num_shifts
        basic_batch_size = batch_size//3 #three kinds of permutations
        # Use vectorized operations
        shifted_temps = [tf.roll(temp[i*batch_size:(i+1)*batch_size], shift=-shifted_integers[i], axis=1) for i in range(num_shifts)]
        #shift permutation
        shift_list = [shifted_temps[i][:basic_batch_size] for i in range(num_shifts)]
        tensor1 = tf.add_n(shift_list)
        shift2_list = [shifted_temps[i][basic_batch_size:2*basic_batch_size] for i in range(num_shifts)]
        tensor2 = tf.add_n(shift2_list)
        width = math.ceil(tensor2.shape[1]/2)
        interleaved = self.interleave_columns(tensor2[:,:width],tensor2[:,width:]) 
        shift3_list = [shifted_temps[i][2*basic_batch_size:] for i in range(num_shifts)]
        tensor3 = tf.add_n(shift3_list)
        permutated = self.inverse_frobenius_automorphism(tensor3, tensor3.shape[1])
        soft_output = soft_output_list[iteration]+(tensor1+interleaved+permutated)/(3*num_shifts)
        soft_output_list[iteration+1] = soft_output
        return soft_output 
    
        
    def interleave_columns(self,tensor1, tensor2):
        """
        Interleave the columns of two tensors into one tensor.
        
        Args:
            tensor1 (tf.Tensor): First tensor of shape (m, n).
            tensor2 (tf.Tensor): Second tensor of shape (m, n) or (m, n-1).
        
        Returns:
            tf.Tensor: Tensor with interleaved columns.
        """
        # Ensure tensors have compatible shapes
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError("Tensors must have the same number of rows")
        
        # Get shapes
        m, n1 = tensor1.shape
        n2 = tensor2.shape[1]
        
        # Interleave columns using stack and reshape
        if n1 == n2:
            # If the number of columns is the same, stack along the third axis and then reshape
            interleaved = tf.reshape(tf.stack([tensor1, tensor2], axis=2), (m, n1 + n2))
        elif n1 == n2 + 1:
            # If tensor1 has one more column than tensor2
            tensor1_split = tf.split(tensor1, [n2, 1], axis=1)
            interleaved = tf.reshape(tf.stack([tensor1_split[0], tensor2], axis=2), (m, n1 + n2 - 1))
            interleaved = tf.concat([interleaved, tensor1_split[1]], axis=1)
        else:
            raise ValueError("tensor1 must have the same number of columns or one more column than tensor2")
        
        return interleaved
    
    def frobenius_automorphism(self,codeword, n):
        """
        Apply the Frobenius automorphism to a binary codeword using the formula (2 * i) % n.
        
        Args:
            codeword (tf.Tensor): Binary codeword of shape (1, n).
            n (int): Length of the codeword.
        
        Returns:
            tf.Tensor: Permuted codeword.
        """
        # Generate positions using the formula (2 * i) % n
        positions = [(2 * i) % n for i in range(n)]
        
        # Create the permutation matrix using the generated positions
        perm_matrix = tf.eye(n, dtype=codeword.dtype)
        perm_matrix = tf.gather(perm_matrix, positions, axis=1)
        
        # Apply the permutation
        permuted_codeword = tf.matmul(codeword, perm_matrix)
        
        return permuted_codeword
    
    def inverse_frobenius_automorphism(self,codeword, n):
        """
        Apply the inverse Frobenius automorphism to a binary codeword using the inverse of (2 * i) % n.
        
        Args:
            codeword (tf.Tensor): Binary codeword of shape (1, n).
            n (int): Length of the codeword.
        
        Returns:
            tf.Tensor: Permuted codeword (original codeword before applying the Frobenius automorphism).
        """
        # Generate positions using the formula (2 * i) % n
        positions = [(2 * i) % n for i in range(n)]
        
        # Generate the inverse positions by reversing the permutation
        inverse_positions = [positions.index(i) for i in range(n)]
        
        # Create the permutation matrix for the inverse positions
        perm_matrix = tf.eye(n, dtype=codeword.dtype)
        perm_matrix = tf.gather(perm_matrix, inverse_positions, axis=1)
        
        # Apply the inverse permutation
        recovered_codeword = tf.matmul(codeword, perm_matrix)
        
        return recovered_codeword
    
    def continue_condition(self,soft_output_list,loss,labels,iteration):
        condition = (iteration < self.num_iterations) 
        return condition
    
    def belief_propagation_iteration(self,soft_output_list,loss,labels, iteration):
        # compute vc
        super_input,shifted_integers = self.aggregate_cyclic_words(soft_output_list[iteration])
        vc_matrix = self.compute_vc(super_input)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix)      
        # get output for this iteration
        soft_output = self.marginalize(cv_matrix, super_input,shifted_integers,labels,iteration,soft_output_list)  
        iteration += 1   
        loss = self.calculation_loss(soft_output,labels,0.)
        return soft_output_list, loss,labels, iteration

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
#save modified data for postprocessing
def save_decoded_data(buffer_inputs,buffer_labels,file_dir):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print(" Data for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    data = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    Data_gen.make_tfrecord(data, out_filename=file_dir)    
    print("Data storing finished!")

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

#postprocessing after first stage training
def postprocess_training(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        _,_,indices= Model.get_eval(soft_output_list,label)
        buffer_inputs_tmp,buffer_labels_tmp = Model.collect_failed_input_output(soft_output_list,label,indices)   
        buffer_inputs.append( buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    return buffer_inputs,buffer_labels
    
#main training process
def training_block(start_info,Model,optimizer,exponential_decay,selected_ds,log_info,restore_info):
    #query of size of input feedings
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list) 
    start_step,train_steps = start_info
    summary_writer,manager_current = log_info
    ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step= restore_info
    batch_index = start_step
    termination_indicator = False
    start_step = start_step%num_counter
    while True:
        for i in range(start_step,num_counter):
                with tf.GradientTape() as tape:
                    inputs = input_list[i]
                    soft_output_list,loss,label = Model(inputs)
                    fer,ber,_= Model.get_eval(soft_output_list,label)
                    grads = tape.gradient(loss,Model.variables)
                grads_and_vars=zip(grads, Model.variables)
                capped_gradients = [(tf.clip_by_norm(grad,1e2), var) for grad, var in grads_and_vars if grad is not None]
                #capped_gradients = [(tf.clip_by_value(grad,-1,1), var) for grad, var in grads_and_vars if grad is not None]
                optimizer.apply_gradients(capped_gradients)
                with summary_writer.as_default():                               # the logger to be used
                  tf.summary.scalar("loss", loss, step=batch_index)
                  tf.summary.scalar("FER", fer, step=batch_index)  # you can also add other indicators below
                  tf.summary.scalar("BER", ber, step=batch_index)  # you can also add other indicators below     
                # log to stdout 
                print_interval = GL.get_map('print_interval')
                record_interval = GL.get_map('record_interval')
                batch_index = batch_index+1
                if batch_index % print_interval== 0 or batch_index == train_steps-1: 
                    tf.print("Step%4d: Lr:%.3f Ls:%.1f FER:%.3f BER:%.4f"%\
                    (batch_index,exponential_decay(batch_index),loss, fer, ber )) 
                if batch_index % record_interval == 0:
                    print("For all layers at the %4d-th step:"%batch_index)
                    manager_current.save(checkpoint_number=batch_index)
                    for variable in Model.variables:
                        print(str(variable.numpy()))  
                    with open(ckpts_dir_par+'values.txt','a+') as f:
                        f.write("For all layers at the %4d-th step:\n"%batch_index)
                        for variable in Model.variables:
                            f.write(variable.name+' '+str(variable.numpy())) 
                        f.write('\n')  
                if batch_index >=min(train_steps,GL.get_map('termination_step')):
                    termination_indicator = True
                    break
                if batch_index%num_counter==0:
                    start_step=0
        if termination_indicator:
            break
    print("Final selected parameters:")
    for weight in  Model.layer.get_weights():
      print(weight)
    return Model          