#!/usr/bin/env bash


##### vaetf


MODEL_NAME=vaetf1


CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -use_cond2lat \
    model-selection \
        -model_type vaetf \
        -model_name ${MODEL_NAME}.pt \
        -model_folder ./Weights/${MODEL_NAME} \
        -save_folder ./Data/inference/model-selection/${MODEL_NAME} \
        -epoch_list 10 20 \
        -decode_algo multinomial \
        -n_samples 100
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &