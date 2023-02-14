#!/usr/bin/env bash 

# GPU_IDX=1

# TOLERANCE=0.20
# SIMILARITY=0.70

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -model_type ctf \
#         -tolerance ${TOLERANCE}   \
#         -similarity ${SIMILARITY} \
#         -start_epoch 1   \
#         -num_epoch 30    \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/ctf_aug-s${SIMILARITY}-t${TOLERANCE} \
#     >train_ctf_s${SIMILARITY}-t${TOLERANCE}.out 2>&1 &


GPU_IDX=1

TOLERANCE=0.20
SIMILARITY=0.70

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -model_type ctf \
        -tolerance ${TOLERANCE}   \
        -similarity ${SIMILARITY} \
        -start_epoch 7  \
        -num_epoch 30    \
        -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/ctf_aug-s0.70-t0.20/model_6.pt \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/ctf_aug-s${SIMILARITY}-t${TOLERANCE} \
    >train_ctf_s${SIMILARITY}-t${TOLERANCE}.out 2>&1 &