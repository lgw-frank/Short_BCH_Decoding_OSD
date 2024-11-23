# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
#from distfit import distfit
  
class Decodering_model(tf.keras.Model):
    def __init__(self,para_list):
        super(Decodering_model,self).__init__()
        self.layer = Decoder_Layer(para_list)
        # if GL.get_map('selected_decoder_type') in ['OMS']:
        #     self.weighting_scalar_list = []
        #     for i in range(10):
        #         self.weighting_scalar_list.append(self.add_weight(name='decoder_half_factor_'+str(i),shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))     
             
    def call(self,inputs,labels): 
        soft_output_list,loss =  self.layer(inputs,labels)
        return soft_output_list,loss 
    def get_eval(self,soft_output,labels):
        soft_output_list = tf.reshape(soft_output,shape=[-1,labels.shape[0],labels.shape[1]])
        tmp = tf.where(soft_output_list>0,0,1)
        err_batch = tf.where(tmp==labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        FER_data = tf.reduce_all(err_sum!=0,0)
        index = tf.where(FER_data)
        BER_data = tf.reduce_sum(err_sum[-1]*tf.cast(FER_data,tf.int32))
        FER = tf.math.count_nonzero(FER_data)/soft_output_list.shape[1]
        BER = BER_data/(soft_output.shape[1]*soft_output_list.shape[2])
        return FER, BER,index   
    def collect_failed_output_selective(self,soft_output_list,labels,index):
        list_length = self.layer.num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])     
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels 
    def mean_variance_matrics(self,outputs,labels):
        plus_output = tf.boolean_mask(outputs,1-labels)
        plus_mean = tf.reduce_mean(plus_output)
        plus_std_variance = tf.math.reduce_std(plus_output)
        
        minus_output = tf.boolean_mask(outputs,labels)
        minus_mean = tf.reduce_mean(minus_output)
        minus_std_variance = tf.math.reduce_std(minus_output)   
        acc =  plus_mean/plus_std_variance-minus_mean/minus_std_variance
        plus_mv = (plus_mean,plus_std_variance)
        minus_mv = (minus_mean,minus_std_variance)
        return acc,plus_mv,minus_mv 
 
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self,para_list):
        super(Decoder_Layer,self).__init__()
        self.num_iterations = GL.get_map('num_iterations')
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.code = GL.get_map('code_parameters')
        self.L  = GL.get_map('weighted_L')
        self.scalar_factor = GL.get_map('scalar_factor')
        self.para_list = para_list
    def build(self, input_shape):          
        if GL.get_map('selected_decoder_type') in ['OMS']:
            self.shared_check_weight = []
            for k in range(GL.get_map('postprocessing_times')):
                self.shared_check_weight.append(self.add_weight(name='decoder_check_normalized factor_'+str(k),shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.para_list[k] )))               
            
          # Code for model call (handles inputs and returns outputs)
    def call(self,inputs,labels):   
        bp_result = self.belief_propagation_op(inputs,labels)
        soft_output_list,loss = bp_result[4],bp_result[5]
        return soft_output_list,loss
       
                
# builds a belief propagation TF graph

    def belief_propagation_op(self,soft_input, labels): 
        
        #loss = self.calculation_loss(soft_input,labels)
        soft_output_list = [soft_input]
        init_value = tf.zeros(soft_input.shape,dtype=tf.float32)
        for _ in range(self.num_iterations):         
            soft_output_list.append(init_value)
        return tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_input, # soft input for this iteration
                labels,
                0, # iteration number
                tf.zeros([soft_input.shape[0],self.code.check_matrix_row,self.code.check_matrix_column],dtype=tf.float32)    ,# cv_matrix
                soft_output_list,  # soft output for this iteration
                0.  #loss value
            ]
            )
          
    # compute messages from variable nodes to check nodes
    def compute_vc(self,cv_matrix, soft_input,iteration):
        normalized_tensor = 1.0
        check_matrix_H = tf.cast(self.code.H,tf.float32)
        if GL.get_map('selected_decoder_type') in ['ADNMS','ANMS','ASNMS']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight_one[iteration])
        if GL.get_map('selected_decoder_type') in ['DNMS','SNMS','NMS']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight_one)
            
        soft_input_weighted = soft_input*normalized_tensor           
        temp = tf.reduce_sum(cv_matrix,1)                        
        temp = temp+soft_input_weighted
        temp = tf.expand_dims(temp,1)
        temp = temp*check_matrix_H
        vc_matrix = temp - cv_matrix
        return vc_matrix  
    # compute messages from check nodes to variable nodes
    def compute_cv(self,vc_matrix,iteration):
        normalized_tensor = 1.0
        check_matrix_H = self.code.H
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
        if GL.get_map('selected_decoder_type') in ['NMS','OMS']:
            normalized_tensor = tf.nn.softplus(self.shared_check_weight[-1])
        
        cv_matrix = normalized_tensor *result_matrix*tf.stop_gradient(result_sign_matrix)         
        return cv_matrix
        

    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input,iteration,soft_output_list):
        normalized_tensor = 1.0
        if GL.get_map('selected_decoder_type') in ['DNMS','SNMS','NMS']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight_two)
        temp = tf.reduce_sum(cv_matrix,1)
        soft_output = temp+normalized_tensor*soft_input
        soft_output_list[iteration+1] = soft_output
    
    def continue_condition(self,soft_input,labels,iteration, cv_matrix, soft_output_list,loss):
        condition = (iteration < self.num_iterations) 
        return condition
    
    
    def mean_variance_estimation(self,outputs):
      abs_output = tf.abs(outputs)
      mean = tf.reduce_mean(abs_output,1,keepdims=True)
      std_variance = tf.math.reduce_std(abs_output,1,keepdims=True)
      return mean,std_variance   


    #common sense definition taking into account all bits of codewords         
    def calculation_loss(self,soft_output,labels,loss):
          #cross entroy
        labels = tf.cast(labels,tf.float32)
        #normalize output
        #soft_output = tf.math.l2_normalize(soft_output,axis=-1)
        CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels))      
        #minimum squared error 
        soft_prob = tf.sigmoid(-soft_output)
        MSE_loss = tf.reduce_sum(tf.square(soft_prob - labels)) 
        new_loss = self.L * CE_loss + (1 - self.L) * MSE_loss*self.scalar_factor
        return new_loss    
    
    def calculation_residual(self,soft_output,labels):
        #cross entroy
        n = soft_output.shape[1]
        labels = tf.cast(labels,tf.float32)
        binary_bpsk = tf.where(labels==0,1.0,-1.0)
        inner_product_coefficient = tf.reduce_sum(soft_output*binary_bpsk,axis=-1,keepdims=True)/n
        residual = soft_output-inner_product_coefficient*binary_bpsk
        return residual   
    def belief_propagation_iteration(self,soft_input, labels, iteration, cv_matrix,soft_output_list,loss):
        # compute vc
        vc_matrix = self.compute_vc(cv_matrix, soft_input,iteration)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix,iteration)      
        # get output for this iteration
        self.marginalize(cv_matrix, soft_input,iteration,soft_output_list) 
        if iteration==2:
            loss = self.calculation_loss(soft_output_list[iteration], labels,loss)
        iteration += 1
        return soft_input, labels, iteration, cv_matrix,soft_output_list,loss



