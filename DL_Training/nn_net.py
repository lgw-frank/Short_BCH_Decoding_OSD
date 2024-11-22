# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:54:10 2023

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
from tensorflow.keras import  layers
from tensorflow import keras 
import os   

class conv_bitwise(keras.Model):
    def __init__(self):
        super(conv_bitwise, self).__init__()   
        code = GL.get_map('code_parameters')
        list_length = GL.get_map('num_iterations')+1
        self.n_dims = code.check_matrix_col
        self.list_length = list_length
        self.H = code.original_H
        #self.attention_layer = SelfAttention(units=32)
    def build(self,input_shape):
        self.cnv_one = layers.Conv1D(filters=2,kernel_size=3,strides=1,\
                                     padding="valid",activation='linear',use_bias=False,name='1st_layer')
        self.cnv_two = layers.Conv1D(1,3,1,padding="valid",activation='linear',use_bias=False,name='2nd_layer')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1,activation='linear', name='Dense')
    def call(self, inputs):
        x = self.cnv_one(inputs) 
        x = self.cnv_two(x)
        x = self.flatten(x)
        outputs = tf.reshape(self.dense(x),[-1,self.n_dims])
        return outputs  
    def preprocessing_inputs(self,input_slice):
        original_input = input_slice[0]
        original_label = input_slice[1]
        file_input_data = tf.reshape(original_input,[-1,self.list_length,self.n_dims]) 
        super_input_list = tf.transpose(file_input_data,perm=[1,0,2])
        distortred_inputs = tf.transpose(file_input_data,perm=[0,2,1])
        labels= original_label[0::self.list_length]
        #preprocessing of input data
        squashed_inputs = tf.reshape(distortred_inputs,[-1,self.list_length,1])
        return squashed_inputs,super_input_list,labels 
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.list_length,1,), dtype='float32', name='input')
        #inputs = tf.reshape(inputs,[-1,self.list_length,1])
        x = self.cnv_one(inputs) 
        x =self.cnv_two(x)
        x = self.cnv_three(x)
        x = self.flatten(x)
        outputs = tf.reshape(self.dense(x),[-1,self.n_dims])
        #x= tf.squeeze(x,axis=-1)   
        #output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        #outputs = tf.reshape(x,shape=[-1,self.n_dims])
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'model_cnn_'+str(self.n_dims)+'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy()) 

class rnn_one(keras.Model):
    def __init__(self):
        super(rnn_one, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_col
        self.list_length = list_length
        self.batch_size = GL.get_map('unit_batch_size')
        self.lstm_layer1 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=True)
        self.lstm_layer2 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=False)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)        
        #self.lstm1 =  keras.layers.Bidirectional(self.lstm_layer1,merge_mode='concat')
        #self.lstm_layer2 = keras.layers.LSTM(int(2*self.n_dims),activation="elu",use_bias=False,dropout=0.0,return_sequences=True)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)
        self.simple_rnn1 = keras.layers.SimpleRNN(8*self.n_dims,activation="linear",dropout=0.0,return_sequences=True,use_bias=True)
        #self.simple_rnn2 = keras.layers.SimpleRNN(2*n,activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_rnn3 = keras.layers.SimpleRNN(self.n_dims,activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        #self.simple_gru1 = keras.layers.GRU(int(4*n),activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_gru1 = keras.layers.GRU(self.n_dims,activation="linear",return_sequences=True,use_bias=False)
        self.simple_gru2 = keras.layers.GRU(self.n_dims,activation="linear",dropout=0.0,return_sequences=False,use_bias=False)
        self.output_layer = keras.layers.Dense(self.n_dims,activation="linear",use_bias=False)
    
    def call(self, inputs):
        x = self.simple_gru1(inputs) 
        x = self.simple_gru2(x)
        outputs= self.output_layer(x)
        return   outputs
    def preprocessing_inputs(self,input_slice):
        reformed_inputs = tf.reshape(input_slice[0],shape=[-1,self.list_length,self.n_dims])
        inputs = input_slice[0][self.list_length-1::self.list_length,:]
        labels = input_slice[1][self.list_length-1::self.list_length,:]
        return reformed_inputs,inputs,labels
    
class rnn_two(keras.Model):
    def __init__(self):
        super(rnn_two, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_col
        self.list_length = list_length
        self.batch_size = GL.get_map('unit_batch_size')
        self.lstm_layer1 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=True)
        self.lstm_layer2 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=False)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)        
        #self.lstm1 =  keras.layers.Bidirectional(self.lstm_layer1,merge_mode='concat')
        #self.lstm_layer2 = keras.layers.LSTM(int(2*self.n_dims),activation="elu",use_bias=False,dropout=0.0,return_sequences=True)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)

        
        self.simple_rnn1 = keras.layers.SimpleRNN(self.n_dims,activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_rnn2 = keras.layers.SimpleRNN(self.n_dims,activation="linear",dropout=0.0,return_sequences=False,use_bias=False)
        #self.simple_gru1 = keras.layers.GRU(int(4*n),activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_gru1 = keras.layers.GRU(self.n_dims,activation="linear",return_sequences=True,use_bias=False)
        self.simple_gru2 = keras.layers.GRU(self.n_dims,activation="linear",dropout=0.0,return_sequences=False,use_bias=False)
        self.output_layer = keras.layers.Dense(self.n_dims,activation="linear",use_bias=False)
    
    def call(self, inputs):
        x = self.simple_rnn1(inputs) 
        x = self.simple_rnn2(x)
        outputs= self.output_layer(x)
        return   outputs
    def preprocessing_inputs(self,input_slice):
        reformed_inputs = tf.reshape(input_slice[0],shape=[-1,self.list_length,self.n_dims])
        inputs = input_slice[0][self.list_length-1::self.list_length,:]
        labels = input_slice[1][self.list_length-1::self.list_length,:]
        return reformed_inputs,inputs,labels
                         
class rnn_three(keras.Model):
    def __init__(self):
        super(rnn_three, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.lstm_layer = keras.layers.LSTM(7,use_bias=True,dropout=0.0,return_sequences=True,unroll=True)   
        self.output_layer = keras.layers.Dense(1,activation="linear",use_bias=True)
        self.rnn_2nd = model_rnn_2nd()
    def build(self,input_shape):
        self.coefficient_list = []
        for i in range(self.list_length):
            self.coefficient_list.append(self.add_weight(name='normalized factor'+str(i),shape=[],trainable=True))    
    def call(self, trim_info_list):
        outputs_list = []
        batch_size = trim_info_list[0].shape[0]//self.list_length
        for i in range(self.n_dims):
            inputs_concat =trim_info_list[i]
            inputs = tf.reshape(inputs_concat,shape=[batch_size,self.list_length,-1,inputs_concat.shape[2]])
            inputs = tf.transpose(inputs,perm=[0,2,1,3])
            inputs = tf.reshape(inputs,[-1,self.list_length,inputs.shape[3]])
            inputs_sub_x = inputs[:,:,1:]
            inputs_sub_y = tf.reshape(inputs[:,:,0],[batch_size,-1,self.list_length])
            x = self.lstm_layer(inputs_sub_x) 
            x= self.output_layer(x)
            x = tf.reshape(x,[batch_size,-1,self.list_length])
            output_element = inputs_sub_y+tf.expand_dims(self.coefficient_list,axis=0)*x
            
            reduce_output = tf.transpose(tf.reduce_mean(output_element,axis=1,keepdims=True),perm=[0,2,1])
            outputs_list.append(reduce_output)   
        final_output = self.rnn_2nd(outputs_list,batch_size)  
        return   final_output
    def preprocessing_inputs(self,input_slice):
        code = GL.get_map('code_parameters')
        check_matrix_H = code.original_H
        original_input = input_slice[0]
        original_label = input_slice[1]
        expanded_input = tf.expand_dims(original_input,axis=1)
        information_matrix = expanded_input*check_matrix_H
        trim_info_list = []
        for i in range(code.check_matrix_column):
            selected_info_col = information_matrix[:,:,i:i+1]
            part1 = information_matrix[:,:,:i]
            part2 = information_matrix[:,:,i+1:]
            new_info_matrix = tf.concat([selected_info_col,part1, part2], axis=-1)
            #assuming constant row weight    
            compressed_info_matrix = tf.reshape(new_info_matrix[new_info_matrix!=0.],\
                                                       shape=[original_input.shape[0],-1,self.row_weight])  
            
            check_matrix_col = tf.reshape(check_matrix_H[:,i],[-1,1])
            trimmed_info_matrix = compressed_info_matrix*tf.cast(check_matrix_col,tf.float32)
            trimmed_info_array = tf.reshape(trimmed_info_matrix[trimmed_info_matrix!=0.],\
                                                 shape=[original_input.shape[0],-1,self.row_weight])
            trim_info_list.append(trimmed_info_array)
        return trim_info_list,original_input[self.list_length-1::self.list_length],original_label[self.list_length-1::self.list_length]
    
class model_rnn_2nd(keras.Model):
    def __init__(self):
        super(model_rnn_2nd, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.lstm_layer = keras.layers.LSTM(self.list_length,use_bias=True,dropout=0.0,return_sequences=False,unroll=True)   
        self.output_layer = keras.layers.Dense(1,activation="linear",use_bias=True)  
    def call(self, input_list,batch_size):
        input_array = tf.reshape(input_list,shape=[-1,batch_size,self.list_length,1])
        input_array = tf.transpose(input_array,perm=[1,0,2,3])
        input_array = tf.reshape(input_array,shape=[-1,self.list_length,1])
        x = self.lstm_layer(input_array) 
        x= self.output_layer(x)
        outputs = tf.reshape(x,shape=[batch_size,-1])
        return  outputs
 