#!/usr/bin/env bash

MODEL_TYPE='mlp'
SIMILARITY=0.70
MLP_STACK=1
GPU_IDX=2
NUM_EPOCH=40

# train

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    main_train.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -load_field \
        -data_name moses \
        -data_path /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -field_path /fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/ \
        -load_scaler \
        -model_type ${MODEL_TYPE} \
        -variational \
    train-2nd \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlp_train_stage2_sim${SIMILARITY}_${MLP_STACK}_test \
        -batch_size 128 \
        -num_epoch ${NUM_EPOCH} \
        -start_epoch 1 \
        -train_verbose \
        -train_stage 2 \
        >fuck.out 2>fuck.err &
    # >mlp_train_sim${SIMILARITY}_${MLP_STACK}.out 2>mlp_train_sim${SIMILARITY}_${MLP_STACK}.err &
