#!/usr/bin/env bash


##### train vae - 1 GPU


MODEL_TYPE=vaetf
MODEL_NAME=${MODEL_TYPE}1

CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python \
    train1.py                         \
        -seed 1                       \
        -lr_WarmUpSteps 12000         \
        -use_cond2lat                 \
        -start_epoch 1                \
        -num_epoch 30                 \
        -batch_size 128               \
        -model_type ${MODEL_TYPE}     \
        -model_folder ./Experiment/${MODEL_NAME} \
    # >train-${MODEL_NAME}.out 2>&1 &


##### train vae - 2 GPUs


# MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}1


# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29501 \
#     train1.py                         \
#         -seed 1                       \
#         -use_cond2lat                 \
#         -start_epoch 1                \
#         -num_epoch 30                 \
#         -batch_size 128               \
#         -model_type ${MODEL_TYPE}     \
#         -model_folder ./Experiment/${MODEL_NAME} \
    # >train-${MODEL_NAME}.out 2>&1 &

