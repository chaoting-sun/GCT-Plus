#!/usr/bin/env bash


##### Our settings

GPU_IDX=0
BENCHMARK=moses
MODEL=pvaetf4
EPOCH=15

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py \
        -use_cond2lat \
    p-sampling \
        -property_list logP tPSA QED \
        -decode_algo multinomial \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -model_type pvaetf \
        -model_name model_${EPOCH}.pt               \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/p-sampling/${MODEL}-${EPOCH} \
        -n_jobs 8 \
        -n_samples 10000 \
    >>${MODEL}_${GPU_IDX}.out 2>&1 & \