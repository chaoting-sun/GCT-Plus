#!/usr/bin/env bash


MODEL_NAME=vaetf1


CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py \
    uc-sampling \
        -model_type vaetf \
        -model_name ${MODEL_NAME}.pt \
        -model_folder ./Weights/vaetf \
        -save_folder ./Data/inference/uc-sampling/${MODEL_NAME} \
        -decode_algo multinomial \
        -n_samples 30000 \
    >>uc-sampling.out 2>&1 &
