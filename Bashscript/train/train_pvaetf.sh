#!/usr/bin/env bash 


##### Training method 1


MODEL_NAME=pvaetf1


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python \
    train.py                          \
        -seed 1                       \
        -use_cond2lat                 \
        -start_epoch 1                \
        -num_epoch 30                 \
        -batch_size 128               \
        -model_type pvaetf            \
        -property_list logP tPSA QED  \
        -model_folder ./Experiment/${MODEL_NAME} \
    # >train-${MODEL_NAME}.out 2>&1 &


##### Training method 2


# MODEL_NAME=pvaetf1


# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python \
#     train1.py \
#         -seed 1                      \
#         -use_cond2lat                \
#         -model_type pvaetf           \
#         -start_epoch 1               \
#         -batch_size 128              \
#         -num_epoch 30                \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#     >train-${MODEL_NAME}.out 2>&1 &