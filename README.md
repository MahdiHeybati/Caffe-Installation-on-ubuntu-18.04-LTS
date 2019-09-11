#  Caffe installation on ubuntu-18.04 LTS, python 2.7, Cuda 10.0, Cudnn 7.6.3 .

### Follow below steps

1- install ubuntu 18.04

2- update and upgrade apt

3- install nvidia driver from > software and updates > additional drivers

4- install cuda 10.0 :

a) install cuda 10.0 deb file for local repository

b) sudo apt update

c) sudo apt install cuda-toolkit-10-0

5- add cuda to path using [here](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-1-on-Ubuntu-19-04-1405/#System-widealternative) or [here](https://www.howtoforge.com/tutorial/how-to-install-nvidia-cuda-on-ubuntu-1804/)

6- install cudnn from source using [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar)

7- fix cudnn symlinks using [here](https://askubuntu.com/questions/1025928/why-do-i-get-sbin-ldconfig-real-usr-local-cuda-lib64-libcudnn-so-7-is-not-a)

8- install caffe dependencies listed in [here](https://medium.com/@atinesh/caffe-installation-on-ubuntu-18-04-lts-python-2-7-8e8c388ce51f) > step 2

9- install caffe with fast-rcnn :

a) git clone fast-rcnn from [here](https://github.com/rbgirshick/py-faster-rcnn) steps 1 and 2

b) install Cython with pip and build cython modules in step 3

c) do these adjustments :

	1) cd caffe-fast-rcnn  
	
	2) git remote add caffe https://github.com/BVLC/caffe.git  
	
	3) git fetch caffe 
	
	4) git merge -X theirs caffe/master
	
	5) remove or comment below line from include/caffe/layers/python_layer.hpp after merging.
	
		```
		self_.attr("phase") = static_cast<int>(this->phase_); 
		```
		
d) create Makefile.config by : 

	```
	cp Makefile.config.example Makefile.config
	```
	
e) copy these configs into Makefile :
	

from below line{

	## Refer to http://caffe.berkeleyvision.org/installation.html
	# Contributions simplifying and improving our build system are welcome!

	# cuDNN acceleration switch (uncomment to build with cuDNN).
	USE_CUDNN := 1

	# CPU-only switch (uncomment to build without GPU support).
	# CPU_ONLY := 1

	# uncomment to disable IO dependencies and corresponding data layers
	# USE_OPENCV := 0
	# USE_LEVELDB := 0
	# USE_LMDB := 0

	# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
	#	You should not set this flag if you will be reading LMDBs with any
	#	possibility of simultaneous read and write
	# ALLOW_LMDB_NOLOCK := 1

	# Uncomment if you're using OpenCV 3
	OPENCV_VERSION := 3

	# To customize your choice of compiler, uncomment and set the following.
	# N.B. the default for Linux is g++ and the default for OSX is clang++
	# CUSTOM_CXX := g++

	# CUDA directory contains bin/ and lib/ directories that we need.
	CUDA_DIR := /usr/local/cuda
	# On Ubuntu 14.04, if cuda tools are installed via
	# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
	# CUDA_DIR := /usr

	# CUDA architecture setting: going with all of them.
	# For CUDA < 6.0, comment the *_50 lines for compatibility.
	CUDA_ARCH := 
			# -gencode arch=compute_20,code=sm_20 \
			# -gencode arch=compute_20,code=sm_21 \
			-gencode arch=compute_30,code=sm_30 \
			-gencode arch=compute_35,code=sm_35 \
			-gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_60,code=sm_60  \
			-gencode arch=compute_61,code=sm_61 

	# BLAS choice:
	# atlas for ATLAS (default)
	# mkl for MKL
	# open for OpenBlas
	BLAS := atlas
	# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
	# Leave commented to accept the defaults for your choice of BLAS
	# (which should work)!
	BLAS_INCLUDE := /usr/local/cuda/targets/x86_64-linux/include
	BLAS_LIB := /usr/local/cuda/targets/x86_64-linux/lib

	# Homebrew puts openblas in a directory that is not on the standard search path
	# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
	# BLAS_LIB := $(shell brew --prefix openblas)/lib

	# This is required only if you will compile the matlab interface.
	# MATLAB directory should contain the mex binary in /bin.
	# MATLAB_DIR := /usr/local
	# MATLAB_DIR := /Applications/MATLAB_R2012b.app

	# NOTE: this is required only if you will compile the python interface.
	# We need to be able to find Python.h and numpy/arrayobject.h.
	PYTHON_INCLUDE := /usr/include/python2.7 \
			/usr/lib/python2.7/dist-packages/numpy/core/include
	# Anaconda Python distribution is quite popular. Include path:
	# Verify anaconda location, sometimes it's in root.
	# ANACONDA_HOME := $(HOME)/anaconda
	# PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
			# $(ANACONDA_HOME)/include/python2.7 \
			# $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \

	# Uncomment to use Python 3 (default is Python 2)
	# PYTHON_LIBRARIES := boost_python3 python3.5m
	# PYTHON_INCLUDE := /usr/include/python3.5m \
	#                 /usr/lib/python3.5/dist-packages/numpy/core/include

	# We need to be able to find libpythonX.X.so or .dylib.
	PYTHON_LIB := /usr/lib
	# PYTHON_LIB := $(ANACONDA_HOME)/lib

	# Homebrew installs numpy in a non standard path (keg only)
	# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
	# PYTHON_LIB += $(shell brew --prefix numpy)/lib

	# Uncomment to support layers written in Python (will link against Python libs)
	WITH_PYTHON_LAYER := 1

	# Whatever else you find you need goes here.
	INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
	LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial

	# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
	# INCLUDE_DIRS += $(shell brew --prefix)/include
	# LIBRARY_DIRS += $(shell brew --prefix)/lib

	# Uncomment to use `pkg-config` to specify OpenCV library paths.
	# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
	USE_PKG_CONFIG := 1

	BUILD_DIR := build
	DISTRIBUTE_DIR := distribute

	# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
	# DEBUG := 1

	# The ID of the GPU that 'make runtest' will use to run unit tests.
	TEST_GPUID := 0

	# enable pretty build (comment to see full commands)
	Q ?= @

} till above line


g) run commands below in order :
	
1) make all -j6 // uses 6 cores to make
	
2) make test -j6
	
	(* if following error occurs do these steps :
	"src/caffe/test/test_smooth_L1_loss_layer.cpp:11:35: fatal error: caffe/vision_layers.hpp"   
	-solution : remove or comment below line from /src/caffe/test/test_smooth_L1_loss_layer.cpp :
		#include "caffe/vision_layers.hpp"
3) make runtest -j6
	
4) make distribute -j6
	
h) follow step 4 from https://github.com/rbgirshick/py-faster-rcnn
