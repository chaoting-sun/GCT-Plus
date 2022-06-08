#!/usr/bin/env bash

python3 -u \
    train.py \
        -n_jobs 2 \
        -variational \
    train-1st \
        -batch_size 16 \
        -num_epoch 40 \
        -train_verbose \