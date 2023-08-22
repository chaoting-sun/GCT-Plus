#!/usr/bin/env bash


##### train sca-vae - 2 GPUs


MODEL_NAME=scavaetf1


CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --master_port 29984 \
    train1.py \
        -seed 1                      \
        -model_type scavaetf         \
        -lr_WarmUpSteps 15000        \
        -use_cond2lat                \
        -use_scaffold                \
        -start_epoch 1               \
        -num_epoch 50                \
        -batch_size 64               \
        -model_folder ./Experiment/${MODEL_NAME} \
    # >train-${MODEL_NAME}.out 2>&1 &


