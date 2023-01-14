#!/usr/bin/env bash 

################ SETTINGS ################
    
# MODEL_TYPE=transformer
# GPU_IDX=2
# BATCH_SIZE=128
# NUM_EPOCH=40
# START_EPOCH=1
# FROM_MOLGCT=False

###########################################

# > train from molgct + freezeDecoderOut

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -use_molgct \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/train_${MODEL_TYPE}_freezeDecoderOut \
#         -use_epoch ${START_EPOCH} \
#     train-1st \
#         -batch_size ${BATCH_SIZE} \
#         -num_epoch ${NUM_EPOCH} \
#     >train_${MODEL_TYPE}_${GPU_IDX}.out 2>&1 &

###########################################

# > train from scratch

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -n_jobs 2 \
#         -model_type transformer \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/train_transformer_fromScratch \
#         -use_epoch 0 \
#     train-1st \
#         -batch_size 128 \
#         -num_epoch 40 \
#     >train_transformer_fromScratch.out 2>&1 &
    

############################################

# > train from scratch
# > original data

GPU_IDX=2

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -tolerance 0.00  \
        -similarity 0.00 \
        -start_epoch 1   \
        -num_epoch 30    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat/transformer-140 \
    >train_transformer-140.out 2>&1 &


# GPU_IDX=0
# TOLERANCE=0
# SIMILARITY=1

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py                        \
#         -tolerance    ${TOLERANCE}  \
#         -similarity   ${SIMILARITY} \
#         -start_epoch  1             \
#         -num_epoch    30            \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer_aug-s${SIMILARITY}-t${TOLERANCE}/   \
#     >>train_aug_t${TOLERANCE}_s${SIMILARITY}.out 2>&1 &
