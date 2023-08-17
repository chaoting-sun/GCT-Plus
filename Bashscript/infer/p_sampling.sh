#!/usr/bin/env bash


##### Our settings


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -use_cond2lat \
    p-sampling \
        -property_list logP tPSA QED \
        -model_type pvaetf \
        -model_name pvaetf1.pt \
        -model_folder ./Weights/pvaetf/ \
        -save_folder ./Data/inference/p-sampling/pvaetf1/ \
        -decode_algo multinomial \
        -n_samples 10000 \
    # >>p_sampling.out 2>&1 & \