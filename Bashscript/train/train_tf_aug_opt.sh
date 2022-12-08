#!/usr/bin/env bash


### train a trained transformer - decoder/out
GPU_IDX=3
MODEL_TYPE=transformer
TOLERENCE=0.01
USE_EPOCH=25
N_EPOCH=40
OPTIM=adam

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -tolerance ${TOLERENCE} \
        -n_jobs 4 \
        -model_type ${MODEL_TYPE} \
        -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_${USE_EPOCH}.pt \
        -optimizer_choice ${OPTIM} \
    train-1st \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer_ep${USE_EPOCH}_aug-decoderout_${OPTIM} \
        -start_epoch $((${USE_EPOCH}+1)) \
        -train_params decoder out \
        -batch_size 128 \
        -num_epoch ${N_EPOCH} \
    >transformer_ep${USE_EPOCH}_aug-decoderout_${OPTIM}.out 2>&1 &
