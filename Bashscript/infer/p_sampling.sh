#!/usr/bin/env bash


##### Our settings


MODEL_NAME=pvaetf1


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -use_cond2lat \
    p-sampling \
        -property_list logP tPSA QED \
        -model_type pvaetf \
        -model_name ${MODEL_NAME}.pt \
        -model_folder ./Weights/pvaetf \
        -save_folder ./Data/inference/p-sampling/${MODEL_NAME} \
        -decode_algo multinomial \
        -n_samples 10000 \
    # >>p_sampling.out 2>&1 & \