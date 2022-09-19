#!/usr/bin/env bash

MODEL_TYPE='att_encoder'
LOSS_FCN=kld
BATCH_SIZE=128

MODEL_VERSION='v1'
SIMILARITY=0.80
GPU_IDX=0
NUM_EPOCH=40
START_EPOCH=1

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -u \
#     main_train.py \
#         -similarity 0.80 \
#         -loss_fcn kld \
#         -n_jobs 2 \
#         -load_field \
#         -load_scaler \
#         -model_type 'att_encoder' \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/atttf_train_stage2_sim0.80_kld_v1 \
#         -batch_size 128 \
#         -num_epoch 2 \
#         -start_epoch 1 \
#         -train_verbose \
#         -train_stage 2

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    main_train.py \
        -similarity ${SIMILARITY} \
        -loss_fcn ${LOSS_FCN} \
        -n_jobs 2 \
        -load_field \
        -load_scaler \
        -model_type ${MODEL_TYPE} \
        -variational \
    train-2nd \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/atttf_train_stage2_sim${SIMILARITY}_${LOSS_FCN}_${MODEL_VERSION} \
        -batch_size ${BATCH_SIZE} \
        -num_epoch ${NUM_EPOCH} \
        -start_epoch ${START_EPOCH} \
        -train_verbose \
        -train_stage 2 \
    >train_${MODEL_TYPE}_sim${SIMILARITY}_${LOSS_FCN}_${MODEL_VERSION}.out \
    2>train_${MODEL_TYPE}_sim${SIMILARITY}_${LOSS_FCN}_${MODEL_VERSION}.err &

# train
# CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     main_train.py \
#         -similarity 1 \
#         -n_jobs 2 \
#         -load_field \
#         -field_path /fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/ \
#         -load_scaler \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim1 \
#         -batch_size 128 \
#         -num_epoch 40 \
#         -starting_epoch 1 \
#         -train_verbose \
#         -train_stage 2 \
#     >train_sim1.out 2>train_sim1.err &


# train
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     main_train.py \
#         -similarity ${SIMILARITY} \
#         -n_jobs 2 \
#         -load_field \
#         -load_scaler \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \
#         -batch_size 128 \
#         -num_epoch ${NUM_EPOCH} \
#         -starting_epoch 1 \
#         -train_verbose \
#         -train_stage 2 \
#     >train_sim${SIMILARITY}_${MLP_STACK}.out 2>train_sim${SIMILARITY}_${MLP_STACK}.err &


