#!/usr/bin/env bash


# python3 train_results.py \
#     -begin_epoch 26 \
#     -end_epoch 39 \
#     -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer-ep26_aug/

python3 train_results.py \
    -begin_epoch 1 \
    -end_epoch 40 \
    -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/

# python3 train_results.py \
#     -begin_epoch 26 \
#     -end_epoch 40 \
#     -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer_ep25_aug-decoderout/

####################################

# OPTIM=adagrad

# python3 train_results.py \
#     -begin_epoch 26 \
#     -end_epoch 33 \
#     -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer_ep25_aug-decoderout_${OPTIM}/