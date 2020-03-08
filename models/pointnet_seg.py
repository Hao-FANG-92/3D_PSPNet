import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tensorflow as tf
import tf_util
from transform_nets import input_transform_net, feature_transform_net
import pyramid_nets

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def placeholder_inputs_weight(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
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
    point_feat = tf.expand_dims(net_transformed, [2])
    print(point_feat)

    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')
    print(global_feat)

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)
    print(concat_feat)

    net = tf_util.conv2d(concat_feat, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, num_classes, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [2]) # BxNxC

    return net, end_points

def get_model_pyramid_fine_tune(pointnet_graph, point_cloud, point_coords_in_voxels, num_scale, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    end_points['transform'] = pointnet_graph.get_tensor_by_name("transform_net2/Reshape_1:0")

    points_feat1 = pointnet_graph.get_tensor_by_name("conv5/Relu:0")
    print("points_feat1:", points_feat1)

    # PYRAMID START #
    # m x n x 1024
    points_feat1 = tf.squeeze(points_feat1, [2])
    print(points_feat1)

    # m x n x (4 x 128 = 512)
    points_feat1_concat = pyramid_nets.pyramid_convert_layer(points_feat1, point_coords_in_voxels, num_scale, [256], "Pyramid_1", bn=True, is_training = is_training, bn_decay = bn_decay)
    print(points_feat1_concat)

    # m x n x 1 x 512
    points_feat1_concat = tf.expand_dims(points_feat1_concat, [2])

    # Concat pyramid global and local features
    points_feat1 = tf.expand_dims(points_feat1, [2])
    point_feat_concat = tf.concat(axis=3, values=[points_feat1, points_feat1_concat])
    # PYRAMID END #


    net = tf_util.conv2d(point_feat_concat, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6_pyramid', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7_pyramid', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8_pyramid', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9_pyramid', bn_decay=bn_decay)

    net = tf_util.conv2d(net, num_classes, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10_pyramid')
    net = tf.squeeze(net, [2]) # BxNxC

    return net, end_points

def get_model_fine_tuing_evaluate(point_cloud, point_coords_in_voxels, num_scale, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
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
    point_feat = tf.expand_dims(net_transformed, [2])
    print(point_feat)

    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # PYRAMID START #
    # m x n x 1024
    net = tf.squeeze(net, [2])
    print(net)

    # m x n x (4 x 128 = 512)
    points_feat1_concat = pyramid_nets.pyramid_convert_layer(net, point_coords_in_voxels, num_scale, [256], "Pyramid_1", bn=True, is_training = is_training, bn_decay = bn_decay)
    print(points_feat1_concat)

    # m x n x 1 x 512
    points_feat1_concat = tf.expand_dims(points_feat1_concat, [2])

    # Concat pyramid global and local features
    net = tf.expand_dims(net, [2])
    point_feat_concat = tf.concat(axis=3, values=[net, points_feat1_concat])
    # PYRAMID END #


    net = tf_util.conv2d(point_feat_concat, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6_pyramid', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7_pyramid', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8_pyramid', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9_pyramid', bn_decay=bn_decay)

    net = tf_util.conv2d(net, num_classes, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10_pyramid')
    net = tf.squeeze(net, [2]) # BxNxC

    return net, end_points

def get_model_multi_pyramid_fine_tune(pointnet_graph, point_cloud, point_coords_in_voxels, num_scale, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    end_points['transform'] = pointnet_graph.get_tensor_by_name("transform_net2/Reshape_1:0")

    points_feat1 = pointnet_graph.get_tensor_by_name("conv5/Relu:0")
    print("points_feat1:", points_feat1)

    # PYRAMID START #
    # m x n x 1024
    points_feat1 = tf.squeeze(points_feat1, [2])
    print(points_feat1)

    # m x n x (4 x 128 = 512)
    points_feat1_concat = pyramid_nets.pyramid_convert_layer(points_feat1, point_coords_in_voxels, num_scale, [256], "Pyramid_1", bn=True, is_training = is_training, bn_decay = bn_decay)
    print(points_feat1_concat)

    # m x n x 1 x 512
    points_feat1_concat = tf.expand_dims(points_feat1_concat, [2])

    # Concat pyramid global and local features
    points_feat1 = tf.expand_dims(points_feat1, [2])
    point_feat_concat = tf.concat(axis=3, values=[points_feat1, points_feat1_concat])
    # PYRAMID END #

    net = tf_util.conv2d(point_feat_concat, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6_pyramid1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7_pyramid1', bn_decay=bn_decay)

    points_feat2= net

    # PYRAMID START #
    # m x n x 1024
    points_feat2 = tf.squeeze(points_feat2, [2])
    print(points_feat2)

    # m x n x (4 x 128 = 512)
    points_feat1_concat2 = pyramid_nets.pyramid_convert_layer(points_feat2, point_coords_in_voxels, num_scale, [128], "Pyramid_2", bn=True, is_training = is_training, bn_decay = bn_decay)
    print(points_feat1_concat2)

    # m x n x 1 x 512
    points_feat1_concat2 = tf.expand_dims(points_feat1_concat2, [2])

    # Concat pyramid global and local features
    points_feat2 = tf.expand_dims(points_feat2, [2])
    point_feat_concat2 = tf.concat(axis=3, values=[points_feat2, points_feat1_concat2])
    # PYRAMID END #


    net = tf_util.conv2d(point_feat_concat2, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6_pyramid2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7_pyramid2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8_pyramid2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9_pyramid2', bn_decay=bn_decay)

    net = tf_util.conv2d(net, num_classes, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10_pyramid')
    net = tf.squeeze(net, [2]) # BxNxC

    return net, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.scalar_summary('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.scalar_summary('mat_loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight

def get_loss(pred, label, smpw, end_points):
    """ pred: BxNxC,
        label: BxN, 
    smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
