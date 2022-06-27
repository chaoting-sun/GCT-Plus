#!/usr/bin/env bash

# python3 -u \
#     train.py \
#         -n_jobs 2 \
#         -variational \
#     train-1st \
#         -batch_size 16 \
#         -num_epoch 40 \
#         -train_verbose \

python3 -u \
    main_train.py \
        -similarity 1 \
        -n_jobs 2 \
        -load_field \
        -data_name moses \
        -field_path ./molGCT \
        -load_scaler \
        -variational \
    train-2nd \
        -save_directory stage2_train \
        -batch_size 64 \
        -num_epoch 40 \
        -train_verbose \
        -train_stage 2 \
        -save_path 'Experiment/mlptf_train_test'
