# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:29:58 2022

@author: lgw
"""
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import tensorflow  as tf
import globalmap as GL
import ms_decoder_dense as Decoder_module
from typing import Any, Dict,Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
       
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
def training_stage(restore_info):
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size)

    start_info = GL.training_setting()
    exponential_decay = GL.optimizer_setting()
    #instance of Model creation   
    Model = Decoder_module.Decoding_model()
    optimizer =  tf.keras.optimizers.Adam(exponential_decay)
   
    # save restoring info
    checkpoint = tf.train.Checkpoint(myAwesomeModel=Model, myAwesomeOptimizer=optimizer)
    logger_info = GL.log_setting(restore_info,checkpoint)
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step] = restore_info  
    if restore_step:
        start_step,ckpt_f = Decoder_module.retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial()
        start_info[0] = start_step
        #model = Model.print_model()
        #print_flops(model)
        #Model.obtain_paras()
    if GL.get_map('loss_process_indicator'):
        Model = Decoder_module.training_block(start_info,Model,optimizer,\
                    exponential_decay,iterator,logger_info,restore_info)
    return Model

def post_process_input(Model):
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size)
    #acquiring erroneous cases with necessary modification or perturbation
    GL.set_map('loss_process_indicator', False)
    buffer_list = Decoder_module.postprocess_training(Model,iterator)
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        file_name = 'bch-allzero-retrain.tfrecord'
    else:
        file_name = 'bch-nonzero-retrain.tfrecord'
    retrain_file_dir = data_dir+file_name
    Decoder_module.save_decoded_data(buffer_list[0],buffer_list[1],retrain_file_dir)
    print("Collecting targeted cases of decoding is finished!")


