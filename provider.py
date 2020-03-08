import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
#if not os.path.exists(DATA_DIR):
#    os.mkdir(DATA_DIR)
#if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#    zipfile = os.path.basename(www)
#    os.system('wget %s; unzip %s' % (www, zipfile))
#    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def findVoxelPosition(data_coord, max_num, num_scale):
    # assert(bin(max_num).count('1') == 1, "The maximum voxel number (%f) should be power of 2!" % max_num)

    #dict_voxel_idx = {}

    ## TODO : reverse data_coord from x, y, z to z, y, x (consisten to D * H * W)
    shift_coord = data_coord - data_coord.min(0)
    # calculate voxel index for the smallest scale
    voxel_size = shift_coord.max(0) / max_num
    grid_size = np.ones((1, 3)) * max_num
    voxel_index = np.floor(shift_coord / voxel_size).astype(np.int)
    # cut the maximum value
    voxel_index[voxel_index == max_num] = max_num - 1

    # store voxel index for each scale
    scale_voxel_id = np.zeros([num_scale, voxel_index.shape(0), 3])
    for i in range(num_scale):
        #dict_voxel_idx['coord_index_' + str(i)] = voxel_index
        scale_voxel_id[i, :, :] = voxel_index
        voxel_index = voxel_index // 2

    #return dict_voxel_idx

    # num_scale x m x n x 3
    return scale_voxel_id

def voxle_3d_id_for_batch_data(batch_data, max_num, num_scale):
    '''
    Args:
        batch_data : m x n x 9
    '''

    batch_size = batch_data.shape[0]
    num_point = batch_data.shape[1]

    batch_data_xyz = batch_data[:,:,6:9]

    # m x n x 3
    batch_data_zyx = batch_data_xyz[:,:,::-1]

    # m x n x 3
    shift_zyx = batch_data_zyx - batch_data_zyx.min(axis = 1, keepdims = True)

    # m x 1 x 3
    voxel_size = shift_zyx.max(axis = 1, keepdims = True) / max_num

    #print(voxel_size[0,0,:])
    #nb = input('Choose a number: ')

    # m x n x 3
    voxel_index = np.floor(shift_zyx / voxel_size).astype(np.int)

    # clip the max voxle size
    voxel_index[voxel_index == max_num] = max_num - 1
    #print(voxel_index[0,0,:])
    #nb2 = input('Choose a number: ')

    grid_size = np.ones((1, 3)) * max_num

    # store voxel index for each scale
    scale_voxel_id = np.zeros([num_scale, batch_size, num_point, 3])
    for i in range(num_scale):
        scale_voxel_id[i, :, :, :] = voxel_index
        voxel_index = voxel_index // 2

    # num_scale x m x n x 3
    return scale_voxel_id[::-1, :, :, :].astype(int) # put largest scale in first raw
    #return scale_voxel_id[::-1, :, :, :].astype(int) # put largest scale in first raw


def voxle_3d_id_for_batch_data_part_seg(batch_data, max_num, num_scale):
    '''
    Args:
        batch_data : m x n x 3
    '''

    batch_size = batch_data.shape[0]
    num_point = batch_data.shape[1]

    # m x n x 3
    batch_data_zyx = batch_data[:,:,::-1]

    # m x n x 3
    shift_zyx = batch_data_zyx - batch_data_zyx.min(axis = 1, keepdims = True)

    # m x 1 x 3
    voxel_size = shift_zyx.max(axis = 1, keepdims = True) / max_num

    #print(voxel_size[0,0,:])
    #nb = input('Choose a number: ')

    # m x n x 3
    voxel_index = np.floor(shift_zyx / voxel_size).astype(np.int)

    # clip the max voxle size
    voxel_index[voxel_index == max_num] = max_num - 1
    #print(voxel_index[0,0,:])
    #nb2 = input('Choose a number: ')

    grid_size = np.ones((1, 3)) * max_num

    # store voxel index for each scale
    scale_voxel_id = np.zeros([num_scale, batch_size, num_point, 3])
    for i in range(num_scale):
        scale_voxel_id[i, :, :, :] = voxel_index
        voxel_index = voxel_index // 2

    # num_scale x m x n x 3
    return scale_voxel_id[::-1, :, :, :].astype(int) # put largest scale in first raw


def voxle_3d_id_for_batch_data_3(batch_data, max_num, num_scale):
    '''
    Args:
        batch_data : m x n x 9
    '''

    batch_size = batch_data.shape[0]
    num_point = batch_data.shape[1]

    batch_data_xyz = batch_data[:,:,0:3]

    # m x n x 3
    batch_data_zyx = batch_data_xyz[:,:,::-1]

    # m x n x 3
    shift_zyx = batch_data_zyx - batch_data_zyx.min(axis = 1, keepdims = True)

    # m x 1 x 3
    voxel_size = shift_zyx.max(axis = 1, keepdims = True) / max_num

    #print(voxel_size[0,0,:])
    #nb = input('Choose a number: ')

    # m x n x 3
    voxel_index = np.floor(shift_zyx / voxel_size).astype(np.int)

    # clip the max voxle size
    voxel_index[voxel_index == max_num] = max_num - 1
    #print(voxel_index[0,0,:])
    #nb2 = input('Choose a number: ')

    grid_size = np.ones((1, 3)) * max_num

    # store voxel index for each scale
    scale_voxel_id = np.zeros([num_scale, batch_size, num_point, 3])
    for i in range(num_scale):
        scale_voxel_id[i, :, :, :] = voxel_index
        voxel_index = voxel_index // 2

    # num_scale x m x n x 3
    return scale_voxel_id[::-1, :, :, :].astype(int) # put largest scale in first raw
