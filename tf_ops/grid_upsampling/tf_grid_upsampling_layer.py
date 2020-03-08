import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

ROOT_DIR = os.path.dirname(BASE_DIR)
ROO_ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROO_ROOT_DIR)

import provider

tf_grid_upsampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grid_upsampling.so'))

def grid_upsampling(in_data, in_grid):
    '''
    Input:
        in_data: (batch_size, num_grids, channels) float32 array
        in_grid: (batch_size, num_points) int32 array, index of grid for each point
    Output:
        out_data: (batch_size, num_points, channels) float32 array, max pooling results on each grid
    '''
    return tf_grid_upsampling_module.grid_upsampling(in_data, in_grid)


@tf.RegisterGradient('GridUpsampling')
def _grid_upsampling_grad(op, grad_out):
    '''
    Input:
        in_data: (batch_size, num_grids, channels) float32 array
        in_grid: (batch_size, num_points) int32 array, index of grid for each point
        grad_out: (batch_size, num_points, channels) float32 array
    Output:
        grad_in: (batch_size, num_grids, channels) float32 array, max pooling results on each grid
    '''
    in_data = op.inputs[0]
    in_grid = op.inputs[1]
    #print(grad_out)
    return [tf_grid_upsampling_module.grid_upsampling_grad(in_data, in_grid, grad_out), None]


if __name__=='__main__':
    import numpy as np
    import random
    import time
    from tensorflow.python.ops.gradient_checker import compute_gradient
    random.seed(100)
    np.random.seed(100)
    
    with tf.device('/gpu:0'):


        in_data_1d = np.asarray([9,10,11,3,4,5,0,0,0,0,0,0,0,0,0,11,10,9,8,7,6,0,0,0,0,0,0,0,0,0])
        #in_data_1d = np.asarray([9,10,11,3,4,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,10,9,8,7,6,])
        in_data_input = np.reshape(in_data_1d,[2,5,3])
        
        in_grid_1d = np.asarray([0,1,0,0,0,1,0,0])
        in_grid_constant = np.reshape(in_grid_1d,[2,4])
        
        #in_data = tf.constant(in_data_input.astype('float32'))
        in_data = tf.Variable(in_data_input.astype('float32'))
        in_grid=tf.constant(in_grid_constant.astype('int32'))

        num_grids = 5

        out_data = grid_upsampling(in_data, in_grid)

        #out_data = in_data
       
        loss = tf.reduce_sum(out_data)
        #loss = 2 * out_data

        test_in_data_input = np.asarray([1,2])
        test_in_data = tf.Variable(test_in_data_input.astype('float32'))
        test_sum = tf.reduce_sum(test_in_data)
        test_loss = test_sum + test_sum
        test_train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(test_loss)
    
        
        '''
        max_num = 8
        num_scale = 4
        
        in_data_constant = np.random.random((24,2048,3))
        in_data = tf.Variable(in_data_constant.astype('float32'))

        in_grid_3d = provider.voxle_3d_id_for_batch_data_part_seg(in_data_constant, max_num, num_scale)

        stride = np.asarray([max_num*max_num, max_num, 1])

        out_data_all_scales = []
        
        loss=0
        scale = 3
        #for scale in range(num_scale):

        in_grid_3d_scale = in_grid_3d[scale, :, :, :]
        in_grid_1d_scale = np.matmul(in_grid_3d_scale, stride)

        num_grids = np.power(np.power(2,scale), 3)

        in_grid=tf.constant(in_grid_1d_scale.astype('int32'))
        
        out_data = grid_upsampling(in_data, in_grid)

        out_data_all_scales.append(out_data)


        loss = tf.reduce_sum(out_data)


        #in_grid_3d_last_scale = in_grid_3d[-1, :, :, :]
        #print(in_grid_3d_last_scale.shape)

        
        #in_grid_1d_last_scale = np.matmul(in_grid_3d_last_scale, stride)
        #print(in_grid_1d_last_scale.shape)
        
        #in_data = tf.Variable(in_data_constant.astype('float32'))
        #in_grid=tf.constant(in_grid_1d_last_scale.astype('int32'))
        
        #out_data, out_pooling_mask = grid_pooling(in_data, in_grid, num_grids)
        '''

        
        
        train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    with tf.Session('') as sess:
        sess.run(tf.global_variables_initializer())
        #now = time.time() 

        #t0=time.time()
        #in_data_value = sess.run(in_data)
        #in_grid_value = sess.run(in_grid)
        #print ("in:", in_data_value)
        #print ("grid:",in_grid_value)

        for i in range(5):
            print(i)
            
            out = sess.run(out_data)
            print ("out:", out)

            in_data_value = sess.run(in_data)
            print ("in_before:", in_data_value)

            trainloss,_ = sess.run([loss,train])
            #trainloss,_,in_data_value,out=sess.run([loss,train,in_data,out_data])

            #print (trainloss)
            #print (time.time() - t0)

            
            in_data_value = sess.run(in_data)
            #in_grid_value = sess.run(in_grid)

            print ("in_after:", in_data_value)
            #print ("grid:",in_grid_value)

            #test_trainloss,_test=sess.run([test_loss,test_train])
            #test_in_data_value = sess.run(test_in_data)
            #print ("in:", test_in_data_value)
   

    
    
