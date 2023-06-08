#!/usr/bin/env bash


BENCHMARK=moses
MODEL_TYPE=vaetf
# MODEL_NAME=vaetf2
# MODEL_FOLDER=/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME}

# --master_port 29502
# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python  \
#     Train.py                          \
#         -use_cond2lat                 \
#         -benchmark ${BENCHMARK}       \
#         -start_epoch 17               \
#         -num_epoch 60                 \
#         -batch_size 128               \
#         -model_type ${MODEL_TYPE}     \
#         -model_folder ${MODEL_FOLDER} \
#     >train-${MODEL_NAME}.out 2>&1 &

MODEL_TYPE=vaetf
MODEL_NAME=${MODEL_TYPE}-warmup12000

CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 nohup python  \
    Train1.py                         \
        -lr_WarmUpSteps 12000         \
        -use_cond2lat                 \
        -benchmark ${BENCHMARK}       \
        -start_epoch 29               \
        -num_epoch 60                 \
        -batch_size 128               \
        -model_type ${MODEL_TYPE}     \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
    >train-${MODEL_NAME}.out 2>&1 &