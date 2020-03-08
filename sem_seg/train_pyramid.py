import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import time
import resource

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *
import pyramid_nets

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path')
FLAGS = parser.parse_args()

print(FLAGS.gpu)
print(FLAGS.log_dir)
print(FLAGS.test_area)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL_PATH = FLAGS.model_path

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train_pyramid.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# voxel size
num_scale = 4
max_num = 8

HOSTNAME = socket.gethostname()

print("base_dir: " + BASE_DIR)
print("file_list: " + os.path.join(BASE_DIR, 'indoor3d_sem_seg_hdf5_data', 'all_files.txt'))

#ALL_FILES = provider.getDataFiles('sem_seg/indoor3d_sem_seg_hdf5_data/all_files.txt')
#room_filelist = [line.rstrip() for line in open('sem_seg/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

ALL_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'indoor3d_sem_seg_hdf5_data', 'all_files.txt'))
room_filelist = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print ([(str(i.name) + '\n') for i in not_initialized_vars])# only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    h5_filename = os.path.join(BASE_DIR, h5_filename)
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

test_area = 'Area_'+str(FLAGS.test_area)
train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            #pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            #loss = get_loss(pred, labels_pl)

            # num_scale x m x n x 3 (voxel coordinates for each point in each scale)
            point_coords_in_voxels = placeholder_inputs_voxel_id(num_scale, BATCH_SIZE, NUM_POINT)
            pred, end_points = get_model_pyramid(pointclouds_pl, point_coords_in_voxels, num_scale, is_training_pl, bn_decay=bn_decay)
            #pred, end_points = get_model_with_pyramid(pointclouds_pl, point_coords_in_voxels, num_scale, is_training_pl, bn_decay=bn_decay)
            #pred, end_points = transfer_learning_with_pyramid(pointclouds_pl, point_coords_in_voxels, num_scale, is_training_pl, bn_decay=bn_decay)
        
            loss = get_loss_pyramid_with_transform_nets(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # pretrained model
        #saver.restore(sess, MODEL_PATH)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'point_3d_voxel_id' : point_coords_in_voxels,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_fine_tuning():
    #with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
    	# Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)


        # load pretrained model
        saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
        saver.restore(sess, MODEL_PATH)

        pointnet_graph = tf.get_default_graph()


        #pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        #is_training_pl = tf.placeholder(tf.bool, shape=())

        pointclouds_pl = pointnet_graph.get_tensor_by_name("Placeholder:0")
        labels_pl = pointnet_graph.get_tensor_by_name("Placeholder_1:0")
        is_training_pl = pointnet_graph.get_tensor_by_name("Placeholder_2:0")
            
        # Note the global_step=batch parameter to minimize. 
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        batch = tf.Variable(0)
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)

        # Get model and loss 
        #pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
        #loss = get_loss(pred, labels_pl)

        # num_scale x m x n x 3 (voxel coordinates for each point in each scale)
        #point_coords_in_voxels = placeholder_inputs_voxel_id(num_scale, BATCH_SIZE, NUM_POINT)
        point_coords_in_voxels = placeholder_inputs_voxel_id(num_scale, BATCH_SIZE, NUM_POINT)
            

        #from tensorflow.python import pywrap_tensorflow
        #reader = pywrap_tensorflow.NewCheckpointReader(MODEL_PATH)
        #var_to_shape_map = reader.get_variable_to_shape_map()

        #for key in sorted(var_to_shape_map):
        #    print("tensor_name: ", key)

        #op = sess.graph.get_operations()
        #for m in op:
        #    print("operators:", m.values())
    

        #print("test_tensor:", pointnet_graph.get_tensor_by_name("conv2/weights:0"))
        #print("points_feat1:", pointnet_graph.get_tensor_by_name("conv5/Relu:0"))
        #print("bn_decay:", bn_decay)


        #pred, end_points = fine_tuning_with_pyramid_3(pointnet_graph, pointclouds_pl, point_coords_in_voxels, num_scale, is_training_pl, bn_decay=bn_decay)
        pred, end_points = fine_tuning_with_pyramid(pointnet_graph, pointclouds_pl, point_coords_in_voxels, num_scale, is_training_pl, bn_decay=bn_decay)


        loss = get_loss_pyramid_with_transform_nets(pred, labels_pl, end_points)
        tf.summary.scalar('loss', loss)

        correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
        tf.summary.scalar('accuracy', accuracy)

        # Get training operator
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, name='fine_tune_adam')
        
        train_op = optimizer.minimize(loss, global_step=batch)

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        #    train_op = optimizer.minimize(loss, global_step=batch)
        

        # Add ops to save and restore all the variables.
        new_saver = tf.train.Saver()
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})
        #initialize_uninitialized(sess)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'point_3d_voxel_id' : point_coords_in_voxels,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            time_start = time.clock()
            train_one_epoch(sess, ops, train_writer)
            time_elapsed = (time.clock() - time_start)

            log_string('total computational time: %f' % (time_elapsed))
            log_string('peak computational momory: %f in Kb' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = new_saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # num_scale x m x n x 3
        current_point_3d_voxel_id = provider.voxle_3d_id_for_batch_data(current_data[start_idx:end_idx, :, :], max_num, num_scale)
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['point_3d_voxel_id']: current_point_3d_voxel_id,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # num_scale x m x n x 3
        current_point_3d_voxel_id = provider.voxle_3d_id_for_batch_data(current_data[start_idx:end_idx, :, :], max_num, num_scale)

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['point_3d_voxel_id']: current_point_3d_voxel_id,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    #train()
    train_fine_tuning()
    LOG_FOUT.close()
