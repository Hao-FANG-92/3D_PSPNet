## Pyramid Scene Parsing Network in 3D: improving semantic segmentation of point clouds with multi-scale contextual information.  *ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 154, 2019*, Fang H., Lafarge F.


We propose a 3D pyramid module to enrich pointwise features with multi-scale contextual information. The goal of our work is not to achieve state-of-the-art performances on the datasets, but to propose a generic module that can be concatenated with any 3D neural network to infer richer pointwise features.

[[paper]](https://hal.inria.fr/hal-02159279/document)



## Overview ##
 
The architecture of our 3d-PSPNet is inspired by the succes of [PSPNet](https://arxiv.org/pdf/1612.01105.pdf) applied on 2D images.

![network.png](./)



## Installation

Please following [PointNet++](https://github.com/charlesq34/pointnet2) to install the corresponding version of *python3.6*, *tensorflow 1.4.0*, and install user defined operator in *tf_ops*.



## Semantic Segmentation ##

### Dataset 

Please following the pipeline used in [PointNet](https://github.com/charlesq34/pointnet/tree/master/sem_seg)

### Training

Once you have downloaded prepared HDF5 files or prepared them by yourself, to start training:

    python train.py --log_dir log6 --test_area 6
    
In default a simple model based on vanilla PointNet is used for training. Area 6 is used for test set.

### Testing

Testing requires download of 3D indoor parsing data and preprocessing with `collect_indoor3d_data.py`

After training, use `batch_inference.py` command to segment rooms in test set. In our work we use 6-fold training that trains 6 models. For model1 , area2-6 are used as train set, area1 is used as test set. For model2, area1,3-6 are used as train set and area2 is used as test set... Note that S3DIS dataset paper uses a different 3-fold training, which was not publicly announced at the time of our work.

For example, to test model6, use command:

    python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump --output_filelist log6/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu

Some OBJ files will be created for prediciton visualization in `log6/dump`.

To evaluate overall segmentation accuracy, we evaluate 6 models on their corresponding test areas and use `eval_iou_accuracy.py` to produce point classification accuracy and IoU as reported in the paper. 


