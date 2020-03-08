## Pyramid Scene Parsing Network in 3D: improving semantic segmentation of point clouds with multi-scale contextual information.  *ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 154, 2019*, Fang H., Lafarge F.


We propose a 3D pyramid module to enrich pointwise features with multi-scale contextual information. The goal of our work is not to achieve state-of-the-art performances on the datasets, but to propose a generic module that can be concatenated with any 3D neural network to infer richer pointwise features.

[[paper]](https://hal.inria.fr/hal-02159279/document)



## Overview ##
 
The architecture of our 3d-PSPNet is inspired by the succes of [PSPNet](https://arxiv.org/pdf/1612.01105.pdf) applied on 2D images.

![Architecture of our model](https://github.com/Hao-FANG-92/3D_PSPNet/blob/master/network.png)



## Installation

Please following [PointNet++](https://github.com/charlesq34/pointnet2) to install the corresponding version of *python3.6*, *tensorflow 1.4.0*, and install user defined operators in *tf_ops*.



## Semantic Segmentation ##

### Dataset 

Please following the pipeline used in [PointNet](https://github.com/charlesq34/pointnet/tree/master/sem_seg).

### Training

First, please train the baseline model from scratch.

    python train.py --log_dir log6 --test_area 6
    
Because our 3d-PSPNet is imposed to incorporate multi-scale contextual information for each point, we conclude a fine-tuning strategy to obtain enriched pointwise feature.
    
    python train_pyramid.py --log_dir log6_pyramid --test_area 6 --model_path log6/model.ckpt
    
Note that the implementation of 3d-PSPNet in *train_pyramid.py* is based on a composition of tensorflow operators. For the efficient issue, we also implement two cuda based tensoflow operators *grid_pooling* and *grid_upsampling*. Users can use them for training by
	
	python train_pyramid_cuda.py --log_dir log6_pyramid_cuda --test_area 6 --model_path log6/model.ckpt


### Testing

Users can evaluate the trained model by

	python batch_inference_pyramid.py --model_path log6_pyramid/model.ckpt --dump_dir log6_pyramid/dump_new --output_filelist log6_pyramid/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu

or by

	python batch_inference_pyramid_cuda.py --model_path log6_pyramid_cuda/model.ckpt --dump_dir log6_pyramid_cuda/dump_new --output_filelist log6_pyramid_cuda/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu


Finally, evaluate overall segmentation accuracy by

	python eval_iou_accuracy.py
	

### Citation

If you find our work useful for your reasearch topic, please cite our paper by

	@article{Fang_jprs19,
	author = {Fang, Hao and Lafarge, Florent},
	title = {{Pyramid scene parsing network in 3D: Improving semantic segmentation of point clouds with multi-scale contextual information}},
	journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
	volume = {154},
	year = {2019},	
	}

### License

 MIT License
 
### Acknowledgement

The main structure of our code is based on [PointNet](https://github.com/charlesq34/pointnet/tree/master/sem_seg).
The cuda implementation of our tf_ops is inspired by [RSNet](https://github.com/qianguih/RSNet).
