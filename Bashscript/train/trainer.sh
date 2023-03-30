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

# GPU_IDX=0
# SIMILARITY=1.00
# TOLERANCE=0.00

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py \
#         -model_type cvaetf \
#         -tolerance ${TOLERANCE}     \
#         -similarity ${SIMILARITY}   \
#         -start_epoch 1     \
#         -num_epoch 30      \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/cvaetf_pad \
#         -pad_to_same_len \
#     >train_cvaetf_s${SIMILARITY}-t${TOLERANCE}.out 2>&1 &


# GPU_IDX=1
# TOLERANCE=0
# SIMILARITY=

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     train.py                        \
#         -tolerance    ${TOLERANCE}  \
#         -similarity   ${SIMILARITY} \
#         -start_epoch  1             \
#         -num_epoch    30            \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer_aug-s${SIMILARITY}-t${TOLERANCE}/   \
#     >>train_aug_t${TOLERANCE}_s${SIMILARITY}.out 2>&1 &

# GPU_IDX=0
# BENCHMARK=chembl_02

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     Train.py \
#         -model_type cvaetf \
#         -benchmark ${BENCHMARK} \
#         -start_epoch 1     \
#         -num_epoch 40      \
#         -max_strlen 100    \
#         -property_list logP \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/cvaetf \
#     >train_cvaetf_s${SIMILARITY}-t${TOLERANCE}.out 2>&1 &


# GPU_IDX=0
# BENCHMARK=moses

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
#     Train.py \
#         -model_type cvaetf \
#         -benchmark ${BENCHMARK} \
#         -start_epoch 1     \
#         -num_epoch 30      \
#         -max_strlen 100    \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/cvaetf_test \

# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

BENCHMARK=moses
SIMILARITY=1.00

# ScaCvaetfV1
# MODELTYPE=scacvaetfv1
# MODEL_NAME=${MODELTYPE}-s${SIMILARITY}
# EPOCH=12

# ScaCvaetfV2
MODELTYPE=scacvaetfv3
MODEL_NAME=${MODELTYPE}-s${SIMILARITY}-revised
START_EPOCH=1

CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun \
    Train.py \
        -benchmark ${BENCHMARK}      \
        -model_type ${MODELTYPE}     \
        -start_epoch ${START_EPOCH}  \
        -num_epoch 50                \
        -max_strlen 80               \
        -property_list logP tPSA QED \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
        -similarity ${SIMILARITY}    \
        -use_scaffold                \
        -batch_size 64               \
    >train-${MODEL_NAME}.out 2>&1 &


# ScaCvaetfV2
# MODELTYPE=scacvaetfv2
# MODEL_NAME=${MODELTYPE}-s${SIMILARITY}-25ep-enum

# CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_LAUNCH_BLOCKING=1 torchrun \
#     Train.py \
#         -benchmark ${BENCHMARK}      \
#         -model_type ${MODELTYPE}     \
#         -start_epoch 31              \
#         -num_epoch 50                \
#         -max_strlen 80               \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
#         -similarity ${SIMILARITY}    \
#         -use_scaffold                \
#         -randomize                   \
    # >train-${MODEL_NAME}.out 2>&1 &