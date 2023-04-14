#!/usr/bin/env bash


BENCHMARK=moses
MODEL_TYPE=vaetf
MODEL_NAME=vaetf-debug
MODEL_FOLDER=/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME}


CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 torchrun --master_port 29502 \
    Train.py                          \
        -use_cond2lat                 \
        -benchmark ${BENCHMARK}       \
        -start_epoch 1                \
        -num_epoch 50                 \
        -batch_size 128               \
        -model_type ${MODEL_TYPE}     \
        -model_folder ${MODEL_FOLDER} \
    # >train-${MODEL_NAME}.out 2>&1 &