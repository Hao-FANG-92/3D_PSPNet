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

tf_grid_pooling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grid_pooling.so'))


def grid_pooling(in_data, in_grid, num_grids):
    '''
    Input:
        in_data: (batch_size, num_points, channel) float32 array
        in_grid: (batch_size, num_points) int32 array, index of grid for each point
        num_grids: const int32, number of total grids (attrs)
    Output:
        out_data: (batch_size, num_grids, channel) float32 array, max pooling results on each grid
        out_pooling_mask: (batch_size, num_grids, channel) int32 array, index of point on each grid with max value
    '''
    return tf_grid_pooling_module.grid_pooling(in_data, in_grid, num_grids)
#ops.NoGradient('GridPooling')


@tf.RegisterGradient('GridPooling')
def _grid_pooling_grad(op, grad_data, grad_pooling_mask):

    #num_grids = op.get_attr('num_grids')
    pooling_mask = op.outputs[1]
    in_data = op.inputs[0]
    
    #grad_list = list(grads)
    #grad_out = grads[0]
    # return gradient of "in_data" with shape (batch_size, num_points, channel)
    return [tf_grid_pooling_module.grid_pooling_grad(in_data, pooling_mask, grad_data), None]




if __name__=='__main__':
    import numpy as np
    import random
    import time
    from tensorflow.python.ops.gradient_checker import compute_gradient
    random.seed(100)
    np.random.seed(100)
    
    with tf.device('/gpu:0'):
     
        
        in_data_1d = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,11,10,9,8,7,6,5,4,3,2,1,0])
        in_data_input = np.reshape(in_data_1d,[2,4,3])
        
        in_grid_1d = np.asarray([0,1,0,0,0,1,0,0])
        in_grid_constant = np.reshape(in_grid_1d,[2,4])
        
        #in_data = tf.constant(in_data_input.astype('float32'))
        in_data = tf.Variable(in_data_input.astype('float32'))
        in_grid=tf.constant(in_grid_constant.astype('int32'))

        num_grids = 5

        out_data, out_pooling_mask = grid_pooling(in_data, in_grid, num_grids)
        #loss = tf.reduce_sum(out_data)

        loss = 2 * out_data
        
        '''
        max_num = 8
        num_scale = 4
        
        in_data_constant = np.random.random((24,2048,3))
        in_data = tf.Variable(in_data_constant.astype('float32'))

        in_grid_3d = provider.voxle_3d_id_for_batch_data_part_seg(in_data_constant, max_num, num_scale)

        stride = np.asarray([max_num*max_num, max_num, 1])

        out_data_all_scales = []
        out_pooling_mask_all_scales = []
        loss=0
        #for scale in range(num_scale):

        scale = 3

        in_grid_3d_scale = in_grid_3d[scale, :, :, :]
        in_grid_1d_scale = np.matmul(in_grid_3d_scale, stride)

        num_grids = np.power(np.power(2,scale), 3)

        in_grid=tf.constant(in_grid_1d_scale.astype('int32'))
        
        out_data, out_pooling_mask = grid_pooling(in_data, in_grid, num_grids)

        out_data_all_scales.append(out_data)
        out_pooling_mask_all_scales.append(out_pooling_mask)

        loss += tf.reduce_sum(out_data)
        


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
       
        t0=time.time()
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
