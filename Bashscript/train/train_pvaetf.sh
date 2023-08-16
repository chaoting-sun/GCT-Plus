#!/usr/bin/env bash 


MODEL_NAME=${MODELTYPE}-QED-test


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python \
    Train.py                          \
        -seed 1                       \
        -use_cond2lat                 \
        -start_epoch 1                \
        -num_epoch 50                 \
        -batch_size 128               \
        -model_type pvaetf            \
        -property_list QED            \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
    # >train-${MODEL_NAME}-${GPU}.out 2>&1 &

