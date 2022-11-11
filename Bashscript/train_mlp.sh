#!/usr/bin/env bash

MODEL_TYPE='mlp'
SIMILARITY=0.70
MLP_STACK=1
GPU_IDX=1
NUM_EPOCH=3

# train

until CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    main_train.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -load_field \
        -field_path /fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/ \
        -load_scaler \
        -model_type ${MODEL_TYPE} \
        -variational \
    train-2nd \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlp_train_stage2_sim${SIMILARITY}_${MLP_STACK}_last \
        -batch_size 128 \
        -num_epoch ${NUM_EPOCH} \
        -start_epoch 1 \
        -train_verbose \
        -train_stage 2 \
    >mlp_train_sim${SIMILARITY}_${MLP_STACK}_last.out 2>mlp_train_sim${SIMILARITY}_${MLP_STACK}_last.err &
do
    echo "Restarting"
    sleep 2
done

# # CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
#     main_train.py \
#         -similarity ${SIMILARITY} \
#         -n_jobs 2 \
#         -load_field \
#         -field_path /fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/ \
#         -load_scaler \
#         -model_type ${MODEL_TYPE} \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlp_train_stage2_sim${SIMILARITY}_${MLP_STACK}_ \
#         -batch_size 128 \
#         -num_epoch ${NUM_EPOCH} \
#         -start_epoch 1 \
#         -train_verbose \
#         -train_stage 2 \
#     # >mlp_train_sim${SIMILARITY}_${MLP_STACK}_.out 2>mlp_train_sim${SIMILARITY}_${MLP_STACK}_.err &
#     # >fuck.out 2>fuck.err &
