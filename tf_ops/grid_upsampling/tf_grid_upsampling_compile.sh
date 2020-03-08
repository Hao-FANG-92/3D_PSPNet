#/bin/bash

#module load conda/5.0.1-python3.6

#source activate tensorflow

#module load cuda/8.0
#module load cudnn/6.0-cuda-8.0

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc tf_grid_upsampling.cu -o tf_grid_upsampling.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 tf_grid_upsampling.cpp tf_grid_upsampling.cu.o -o tf_grid_upsampling.so -shared -fPIC -I $TF_INC -I /usr/local/cuda/include -I $TF_INC/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
