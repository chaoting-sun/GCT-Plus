#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
    uc-sampling \
        -model_type vaetf \
        -model_name vaetf1.pt \
        -model_folder ./Weights/vaetf/ \
        -save_folder ./Data/inference/uc-sampling/vaetf1 \
        -decode_algo multinomial \
        -n_samples 30000 \
    # >>uc-sampling.out 2>&1 &
