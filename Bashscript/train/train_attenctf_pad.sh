#!/usr/bin/env bash 

GPU_IDX=3

TOLERANCE=0.20
SIMILARITY=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -model_type attenctf        \
        -batch_size 64              \
        -tolerance ${TOLERANCE}     \
        -similarity ${SIMILARITY}   \
        -start_epoch 1              \
        -num_epoch 10               \
        -train_params rotator       \
        -use_cvaetf_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/ctf_pad/model_10.pt \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/attenctf_pad_dconds_molgpt_aug-s${SIMILARITY}-t${TOLERANCE} \
        -pad_to_same_len \
    >train_attenctf_pad_gpt_dconds.out 2>&1 &