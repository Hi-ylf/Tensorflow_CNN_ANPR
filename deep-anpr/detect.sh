#!/bin/sh
gpuid=$1 # 设置使用哪个一个GPU，目前只支持单GPU
export CUDA_VISIBLE_DEVICES=$gpuid
#export PATH="/usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so"
export PATH="/usr/local/cuda-8.0/"

./detect.py 1.jpg weights.npz out.jpg
