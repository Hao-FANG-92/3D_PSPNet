#!/bin/sh

module load conda/5.0.1-python3.6

source activate tensorflow

module load cuda/8.0
module load cudnn/6.0-cuda-8.0


#python $1 $2 $3 $4 $5 $6 $7
python tf_grouping_op_test.py
#"$@"