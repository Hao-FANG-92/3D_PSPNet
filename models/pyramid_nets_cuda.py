import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
import tf_util

from grid_pooling import tf_grid_pooling_layer
from grid_upsampling import tf_grid_upsampling_layer

def pyramid_net_cuda(point_features, point_coords_in_voxels, num_scale, num_units, scope, bn = None, is_training = None, bn_decay = None):
    '''
    Args: 
        point_features : m x n x f, global features for each point
        point_coords_in_voxels : num_scale x m x n x 3 (voxel coordinates for each point in each scale)
        num_scale : number of scales in pyramid nets
        num_units : list of number of hidden units
    '''

    batch_size = point_features.get_shape()[0].value
    num_point = point_features.get_shape()[1].value

    list_scale_point_features = []
    
    for i in range(num_scale):
        num_voxel = pow(2, i)
        
        basic_index = np.ones((batch_size, 3)) * [num_voxel * num_voxel, num_voxel, 1]
        basic_index = basic_index.astype(np.int32).reshape((batch_size, 3, 1)) # m x 3 x 1

        # m x n x 1 : 1-D voxel index for each point -> 1-d_id = z_id x (H x W) + y_id x W + x_id
        flattens = tf.matmul(point_coords_in_voxels[i, :, :, :], basic_index)
        flat_1d = tf.squeeze(flattens, [2])  # m x n
        print (flat_1d)

        num_grids = np.power(np.power(2, i), 3)

        # m x num_grids x f && m x num_grids x f
        voxels_features, out_pooling_mask_scale = tf_grid_pooling_layer.grid_pooling(point_features, flat_1d, num_grids)

        with tf.variable_scope(scope + '_' + str(i)):
    
            # Fully connected layer from m x num_grids x f to m x num_grids x num_units
            for num_unit in num_units:

                #voxels_features = tf_util.conv3d(voxels_features, num_unit, [1,1,1], scope='MLP_'+str(num_unit), stride=[1,1,1], padding='SAME', bn=bn, bn_decay=bn_decay, is_training=is_training)

                voxels_features = tf.layers.dense(voxels_features, num_unit, tf.nn.relu, name='dense_'+str(i) + '_' + str(num_unit))
                #print(voxels_features)

                #voxels_features = tf.layers.dense(voxels_features, 256, tf.nn.relu, name='dense_'+str(i) + '_' + str(256) + '_2')
                #voxels_features = tf.layers.dense(voxels_features, 128, tf.nn.relu, name='dense_'+str(i) + '_' + str(128) + '_1')
                #voxels_features = tf.layers.dense(voxels_features, 64, tf.nn.relu, name='dense_'+str(i) + '_' + str(64) + '_2')
                #voxels_features = tf.layers.dense(voxels_features, 128, tf.nn.relu, name='dense_'+str(i) + '_' + str(128) + '_2')
                #voxels_features = tf.layers.dense(voxels_features, num_unit, tf.nn.relu, name='dense_'+str(i) + '_' + str(num_unit) + '_2')

                #bn = False
                if bn:
                    voxels_features = tf.layers.batch_normalization(voxels_features, name='batch_norm'+str(i) + '_' + str(num_unit))
                #    voxels_features = tf.layers.batch_normalization(voxels_features, name='batch_norm'+str(i) + '_' + str(num_unit), training=is_training)

                #    voxels_features = tf_util.batch_norm_for_conv1d(voxels_features, is_training,
                #                      bn_decay=bn_decay, scope='bn' + str(i) + '_' + str(num_unit))
                    #print(voxels_features)

            #dense_voxels_features2 = tf.layers.dense(dense_voxels_features1, num_units[1], tf.nn.relu, name='dense_'+str(1))
            #print(dense_voxels_features2)

            #if bn:
            #    dense_voxels_features2 = tf.layers.BatchNormalization(dense_voxels_features2, name='batch_norm', fused=True)

        # upsampling : m x n x num_units[-1]
        umsampled_point_features = tf_grid_upsampling_layer.grid_upsampling(voxels_features, flat_1d)

        list_scale_point_features.append(umsampled_point_features)

    # 3D Tensor: m x n x (num_scale x num_units[-1])
    scale_point_features = tf.concat(list_scale_point_features, axis = 2)

    return scale_point_features
