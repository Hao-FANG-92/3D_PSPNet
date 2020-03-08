import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def pyramid_convert_layer(point_features, point_coords_in_voxels, num_scale, num_units, scope, bn = None, is_training = None, bn_decay = None):
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
    #for i in range(1, num_scale):
    for i in range(num_scale):
        num_voxel = pow(2, i)
        
        basic_index = np.ones((batch_size, 3)) * [num_voxel * num_voxel, num_voxel, 1]
        basic_index = basic_index.astype(np.int32).reshape((batch_size, 3, 1)) # m x 3 x 1

        # m x n x 1 : 1-D voxel index for each point -> 1-d_id = z_id x (H x W) + y_id x W + x_id
        flattens = tf.matmul(point_coords_in_voxels[i, :, :, :], basic_index)
        print(flattens)

        # m x D x H x W x f -> max pooling in each voxel
        voxels_features = tf.map_fn(fn = lambda element: pick_max_features(element[0], element[1], num_voxel), elems = (point_features, flattens), dtype = (tf.float32))
        print(voxels_features)

        with tf.variable_scope(scope + '_' + str(i)):
            # convolutinal middle layers
            #temp_conv = conv3d()

            # Fully connected layer from m x D x H x W x f to m x D x H x W x num_units
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
        umsampled_point_features = tf.map_fn(fn = lambda element: upsamling_point_features(element[0], element[1]), elems = (point_coords_in_voxels[i, :, :, :], voxels_features), dtype=tf.float32)

        list_scale_point_features.append(umsampled_point_features)

    # 3D Tensor: m x n x (num_scale x num_units[-1])
    scale_point_features = tf.concat(list_scale_point_features, axis = 2)
    #scale_point_features = tf.reshape(scale_point_features, [num_scale, batch_size, num_point, num_units])

    return scale_point_features


def pyramid_convert_layer_hao(point_features, point_coords_in_voxels, num_scale, num_units, bn = None, is_training = None, bn_decay = None):
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
    #for i in range(1, num_scale):
    for i in range(num_scale):
        num_voxel = pow(2, i)
        
        basic_index = np.ones((batch_size, 3)) * [num_voxel * num_voxel, num_voxel, 1]
        basic_index = basic_index.astype(np.int32).reshape((batch_size, 3, 1)) # m x 3 x 1

        # m x n x 1 : 1-D voxel index for each point -> 1-d_id = z_id x (H x W) + y_id x W + x_id
        flattens = tf.matmul(point_coords_in_voxels[i, :, :, :], basic_index)
        print(flattens)

        # m x D x H x W x f -> max pooling in each voxel
        voxels_features = tf.map_fn(fn = lambda element: pick_max_features(element[0], element[1], num_voxel), elems = (point_features, flattens), dtype = (tf.float32))
        print(voxels_features)

        with tf.variable_scope('Pyramid_' + str(i)):
            # convolutinal middle layers
            #temp_conv = conv3d()

            # Fully connected layer from m x D x H x W x f to m x D x H x W x num_units
            for num_unit in num_units:

                voxels_features = tf.layers.dense(voxels_features, num_unit, tf.nn.relu, name='dense_'+str(i) + '_' + str(num_unit))
                #print(voxels_features)

                if bn:
                    voxels_features = tf.layers.batch_normalization(voxels_features, name='batch_norm'+str(i) + '_' + str(num_unit))
                    #print(voxels_features)

            #dense_voxels_features2 = tf.layers.dense(dense_voxels_features1, num_units[1], tf.nn.relu, name='dense_'+str(1))
            #print(dense_voxels_features2)

            #if bn:
            #    dense_voxels_features2 = tf.layers.BatchNormalization(dense_voxels_features2, name='batch_norm', fused=True)

        # upsampling : m x n x num_units[-1]
        umsampled_point_features = tf.map_fn(fn = lambda element: upsamling_point_features(element[0], element[1]), elems = (point_coords_in_voxels[i, :, :, :], voxels_features), dtype=tf.float32)

        list_scale_point_features.append(umsampled_point_features)

    # 3D Tensor: m x n x (num_scale x num_units[-1])
    scale_point_features = tf.concat(list_scale_point_features, axis = 2)
    #scale_point_features = tf.reshape(scale_point_features, [num_scale, batch_size, num_point, num_units])

    return scale_point_features




def pick_max_features(feature, flatten, num_voxel):
    '''
    Args:
        feature : n x f
        flatten : n x 1 -> 1-d voxel id for each point
    '''
    
    # filtered: k -> unique voxel id 
    # idx : n -> indices of each point in filtered
    filtered, idx = tf.unique(tf.squeeze(flatten))

    # k x f -> max pooling in each unique voxel
    updated_features = tf.unsorted_segment_max(feature, idx, tf.shape(filtered)[0])
    #updated_features = tf.unsorted_segment_sum(feature, idx, tf.shape(filtered)[0])
    print(updated_features)

    # k x 3 -> 3-d voxel id for each unique voxel
    updated_indices = tf.map_fn(fn = lambda i: reverse(i, num_voxel), elems = filtered)
    print(updated_indices)

    num_features = updated_features.shape[-1]

    # D x H x W x f
    voxels = tf.scatter_nd(updated_indices, 
                           updated_features, 
                           tf.TensorShape([num_voxel, num_voxel, num_voxel, num_features]))

    return voxels


def reverse(index, num_voxel):
    """Map from 1-D to 3-D """
    x = index // (num_voxel * num_voxel)
    y = (index - x * num_voxel * num_voxel) // num_voxel
    z = index - x * num_voxel * num_voxel - y * num_voxel
    return tf.stack([x, y, z], -1) 


def upsamling_point_features(points_voxel_id, voxel_features):
    '''
    Args:
        points_voxel_id : n x 3 -> 3-d voxel id for each point
        voxel_featuers : D x H x W x f
    '''

    # n x f -> point features
    return tf.gather_nd(voxel_features, points_voxel_id)
