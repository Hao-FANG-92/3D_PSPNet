
#!/bin/sh

#module load conda/5.0.1-python3.6

#source activate tensorflow

#module load cuda/8.0
#module load cudnn/6.0-cuda-8.0

# Define the compiler
CC := g++

# Read Tensorflow paths
TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


#/misc/opt/cuda/8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# Is the Tensorflow version >= 1.4?
TF_VERSION_GTE_1_4 := $(shell expr `python -c 'import tensorflow as tf; print(tf.__version__)' | cut -f1,2 -d.` \>= 1.4)

# Flags required for all cases
CFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC -I$(TF_INC) -O2 

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

# Set some more flags if the Tensorflow version is >= 1.4
ifeq ($(TF_VERSION_GTE_1_4), 1)
    CFLAGS += -I$(TF_INC)/external/nsync/public
    CFLAGS += -I/misc/opt-7.4/cuda/8.0/include
	LDFLAGS := -L$(TF_LIB) -ltensorflow_framework
else
	LDFLAGS :=
endif

# Define build targets
.PHONY: all clean

tf_grouping_so.so: tf_grouping.cpp
	$(CC) $(CFLAGS) -o tf_grouping_so.so tf_grouping.cpp $(LDFLAGS)

clean:
	$(RM) tf_grouping_so.so

all: tf_grouping_so.so
