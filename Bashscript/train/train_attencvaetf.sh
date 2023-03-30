#!/usr/bin/env bash

# GPU_IDX=2

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
#     train.py \
#         -model_type attencvaetf \
#         -tolerance 0.20  \
#         -similarity 0.80 \
#         -use_cvaetf_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_25.pt \
#         -start_epoch 26   \
#         -num_epoch 30    \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/attencvaetf_selftf \
#     # >train_attencvaetf_selftf.out 2>&1 &


BENCHMARK=moses
SIMILARITY=0.70

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_LAUNCH_BLOCKING=1 nohup torchrun \
    Train.py \
        -model_type attencvaetf \
        -benchmark ${BENCHMARK} \
        -start_epoch 3     \
        -num_epoch 20      \
        -max_strlen 80     \
        -property_list logP tPSA QED \
        -original_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/cvaetf/model_25.pt \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/attencvaetf-z-s${SIMILARITY} \
        -similarity_threshold ${SIMILARITY} \
    >train_attencvaetf-s${SIMILARITY}.out 2>&1 &