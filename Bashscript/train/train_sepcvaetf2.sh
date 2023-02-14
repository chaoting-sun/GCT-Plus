#!/usr/bin/env bash

GPU_IDX=1

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -model_type sepcvaetf2 \
        -tolerance 0.20  \
        -similarity 0.60 \
        -use_cvaetf_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_25.pt \
        -start_epoch 26   \
        -num_epoch 30    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/sepcvaetf2 \
    >train_sepcvaetf2.out 2>&1 &