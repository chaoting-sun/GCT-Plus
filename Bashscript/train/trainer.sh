#!/usr/bin/env bash


############################################

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

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup torchrun \
#     Train.py \
#         -use_cond2lat                \
#         -model_type cvaetf \
#         -benchmark ${BENCHMARK} \
#         -start_epoch 1     \
#         -num_epoch 30      \
#         -property_list logP tPSA QED \
#         -batch_size 128               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/cvaetf2+ \
# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# SIMILARITY=1.00

# # ScaCvaetfV1
# # MODELTYPE=scacvaetfv1
# # MODEL_NAME=${MODELTYPE}-s${SIMILARITY}
# # EPOCH=12

# MODELTYPE=scacvaetfv3
# MODEL_NAME=${MODELTYPE}-warmup15000
# START_EPOCH=1
# BENCHMARK=moses

# # --master_port 29502

# # CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python \
# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29990 \
#     Train1.py \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -benchmark ${BENCHMARK}      \
#         -model_type ${MODELTYPE}     \
#         -start_epoch ${START_EPOCH}  \
#         -num_epoch 50                \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#     >train-${MODEL_NAME}.out 2>&1 &


# MODELTYPE=scavaetf
# MODEL_NAME=${MODELTYPE}2-warmup15000
# START_EPOCH=1
# BENCHMARK=moses

# # CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python \
# CUDA_VISIBLE_DEVICES=2,3 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29984 \
#     Train1.py \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -benchmark ${BENCHMARK}      \
#         -model_type ${MODELTYPE}     \
#         -start_epoch ${START_EPOCH}  \
#         -num_epoch 50                \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#     >train-${MODEL_NAME}.out 2>&1 &


# MODELTYPE=scacvaetfv3
# MODEL_NAME=${MODELTYPE}3-beta0.01-warmup15000_molgpt
# START_EPOCH=8
# BENCHMARK=moses

# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29903 \
#     Train1.py                        \
#         -seed 1000                   \
#         -KLA_inc_beta 0.01           \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -benchmark ${BENCHMARK}      \
#         -model_type ${MODELTYPE}     \
#         -start_epoch ${START_EPOCH}  \
#         -num_epoch 50                \
#         -property_list logP tPSA SAS \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#         -randomize_prob 0.5          \
#     >train-${MODEL_NAME}.out 2>&1 &


##### pscavaetf

# MODELTYPE=pscavaetf
# MODEL_NAME=pscavaetf-r0.5
# START_EPOCH=11

# CUDA_VISIBLE_DEVICES=2,3 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29907 \
#     Train1.py \
#         -seed 1000                   \
#         -KLA_inc_beta 0.01           \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -benchmark moses             \
#         -model_type pscavaetf        \
#         -start_epoch ${START_EPOCH}  \
#         -num_epoch 20                \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#         -randomize_prob 0.5          \
#     >train-${MODEL_NAME}.out 2>&1 &


##### pvaetf

# GPU_IDX=2
# MODEL_NAME=pvaetf1

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python \
#     Train1.py \
#         -seed 1000                   \
#         -use_cond2lat                \
#         -benchmark moses             \
#         -model_type pvaetf           \
#         -start_epoch 1               \
#         -batch_size 128              \
#         -num_epoch 30                \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#     >train-${MODEL_NAME}-${GPU_IDX}.out 2>&1 &