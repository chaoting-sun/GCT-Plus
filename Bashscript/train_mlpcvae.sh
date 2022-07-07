#!/usr/bin/env bash


# train
# CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     main_train.py \
#         -similarity 1 \
#         -n_jobs 2 \
#         -load_field \
#         -data_name moses \
#         -data_path /fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim1/ \
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

SIMILARITY=0.70
MLP_STACK=3
GPU_IDX=2

# train
# CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
    main_train.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -load_field \
        -data_name moses \
        -data_path /fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim${SIMILARITY}/ \
        -field_path /fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/ \
        -load_scaler \
        -variational \
    train-2nd \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \
        -batch_size 128 \
        -num_epoch 40 \
        -starting_epoch 1 \
        -train_verbose \
        -train_stage 2 \
    >train_sim${SIMILARITY}_${MLP_STACK}.out 2>train_sim${SIMILARITY}_${MLP_STACK}.err &