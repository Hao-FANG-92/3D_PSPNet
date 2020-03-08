import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

sys.path.append(os.path.join(BASE_DIR, '../models'))
import pyramid_nets

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


def get_model_fine_tuning(pointnet_graph, point_cloud, point_coords_in_voxels, num_scale, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points['transform'] = pointnet_graph.get_tensor_by_name("transform_net2/Reshape_1:0")

    points_feat1 = pointnet_graph.get_tensor_by_name("conv5/Relu:0")
    print("points_feat1:", points_feat1)
    #out_max = tf_util.max_pool2d(out5, [num_point,1], padding='VALID', scope='maxpool')

    # PYRAMID START #
    # m x n x 1024
    points_feat1 = tf.squeeze(points_feat1, [2])
    print(points_feat1)


    # m x n x (4 x 128 = 512)
    points_feat1_concat = pyramid_nets.pyramid_convert_layer(points_feat1, point_coords_in_voxels, num_scale, [256], bn=True, is_training = is_training, bn_decay = bn_decay)
    print(points_feat1_concat)

    # m x n x 1 x 512
    points_feat1_concat = tf.expand_dims(points_feat1_concat, [2])

    # Concat pyramid global and local features
    points_feat1 = tf.expand_dims(points_feat1, [2])
    point_feat_concat = tf.concat(axis=3, values=[points_feat1, points_feat1_concat])
    # PYRAMID END #

    # Symmetric function: max pooling
    # m x 1 x 1 x 2048
    net = tf_util.max_pool2d(point_feat_concat, [num_point,1], padding='VALID', scope='pyramid_maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='pyramid_fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='pyramid_dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='pyramid_fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='pyramid_dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='pyramid_fc3')

    return net, end_points

def get_model_fine_tuning_test(point_cloud, point_coords_in_voxels, num_scale, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    # PYRAMID START #
    # m x n x 1024
    points_feat1 = tf.squeeze(points_feat1, [2])
    print(points_feat1)


    # m x n x (4 x 128 = 512)
    points_feat1_concat = pyramid_nets.pyramid_convert_layer(points_feat1, point_coords_in_voxels, num_scale, [256], bn=True, is_training = is_training, bn_decay = bn_decay)
    print(points_feat1_concat)

    # m x n x 1 x 512
    points_feat1_concat = tf.expand_dims(points_feat1_concat, [2])

    # Concat pyramid global and local features
    points_feat1 = tf.expand_dims(points_feat1, [2])
    point_feat_concat = tf.concat(axis=3, values=[points_feat1, points_feat1_concat])
    # PYRAMID END #

    # Symmetric function: max pooling
    #net = tf_util.max_pool2d(point_feat_concat, [num_point,1], padding='VALID', scope='pyramid_maxpool')
    net = tf_util.avg_pool2d(point_feat_concat, [num_point,1], padding='VALID', scope='pyramid_maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='pyramid_fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='pyramid_dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='pyramid_fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='pyramid_dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='pyramid_fc3')


    return net, end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)

        
