#!/usr/bin/env bash


### constants
GPU_IDX=1
MODEL_TYPE=transformer


### train a trained transformer - decoder/out
# TOLERENCE=0.01
# USE_EPOCH=25
# N_EPOCH=40

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -tolerance ${TOLERENCE} \
#         -n_jobs 4 \
#         -model_type transformer \
#         -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_${USE_EPOCH}.pt \
#     train-1st \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer-ep${USE_EPOCH}_aug \
#         -start_epoch $((${USE_EPOCH}+1)) \
#         -train_params decoder out \
#         -batch_size 128 \
#         -num_epoch ${N_EPOCH} \
#     >train_tf_aug_ep${USE_EPOCH}_tol${TOLERENCE}.out 2>&1 &


### train a trained transformer - all
TOLERENCE=0.01
USE_EPOCH=25
N_EPOCH=40

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -tolerance ${TOLERENCE} \
        -n_jobs 4 \
        -model_type transformer \
        -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_${USE_EPOCH}.pt \
    train-1st \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer-ep${USE_EPOCH}_all_aug \
        -start_epoch $((${USE_EPOCH}+1)) \
        -batch_size 128 \
        -num_epoch ${N_EPOCH} \
    >train_tf-ep${USE_EPOCH}_all_aug_tol${TOLERENCE}.out 2>&1 &


### train an untrained transformer
# TOLERENCE=0.01
# N_EPOCH=40

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -tolerance ${TOLERENCE} \
#         -n_jobs ${N_JOBS} \
#         -model_type transformer \
#         -use_molgct \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer_aug_ep0 \
#     train-1st \
#         -batch_size 128 \
#         -num_epoch ${N_EPOCH} \
#     >train_tf_fromScratch_aug_tol${TOLERENCE}.out 2>&1 &


### train molgct - decoder/out
# TOLERENCE=0.01
# USE_EPOCH=30
# N_EPOCH=30

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -tolerance ${TOLERENCE} \
#         -n_jobs ${N_JOBS} \
#         -model_type transformer \
#         -use_molgct \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/molgct_decoderout_aug_ep${USE_EPOCH} \
#         -use_epoch ${USE_EPOCH} \
#     train-1st \
#         -batch_size ${BATCH_SIZE} \
#         -num_epoch ${N_EPOCH} \
#     >train_molgctDecoderOut_aug_ep${USE_EPOCH}_tol${TOLERENCE}.out 2>&1 &


### train molgct - all
# TOLERENCE=0.01
# USE_EPOCH=30
# N_EPOCH=40

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -tolerance ${TOLERENCE} \
#         -n_jobs ${N_JOBS} \
#         -model_type transformer \
#         -use_molgct \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/molgct_all_aug_ep${USE_EPOCH} \
#         -use_epoch ${USE_EPOCH} \
#     train-1st \
#         -batch_size ${BATCH_SIZE} \
#         -num_epoch ${N_EPOCH} \
#     >train_molgctAll_aug_ep${USE_EPOCH}_tol${TOLERENCE}.out 2>&1 &
