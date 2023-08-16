#!/usr/bin/env bash


##### train vae - 1 GPU


# MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}-test

# CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python \
#     Train1.py                         \
#         -seed 1                       \
#         -lr_WarmUpSteps 12000         \
#         -use_cond2lat                 \
#         -start_epoch 2                \
#         -num_epoch 30                 \
#         -batch_size 128               \
#         -model_type ${MODEL_TYPE}     \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
    # >train-${MODEL_NAME}.out 2>&1 &


##### train p-vae - 1 GPU


MODEL_NAME=pvaetf4


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python \
    Train1.py \
        -seed 4                      \
        -use_cond2lat                \
        -model_type pvaetf           \
        -start_epoch 1               \
        -batch_size 128              \
        -num_epoch 30                \
        -property_list logP tPSA QED \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
    >train-${MODEL_NAME}.out 2>&1 &


##### train sca-vae - 2 GPUs


# MODEL_NAME=scavaetf-test


# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29984 \
#     Train1.py \
#         -seed 1                      \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -model_type scavaetf         \
#         -start_epoch 1               \
#         -num_epoch 50                \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#     >train-${MODEL_NAME}.out 2>&1 &


##### train psca-vae - 2 GPUs


# MODEL_NAME=${MODELTYPE}-test


# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --master_port 29990 \
#     Train1.py \
#         -seed 1                      \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -model_type pscavaetf        \
#         -start_epoch 1               \
#         -num_epoch 50                \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#     # >train-${MODEL_NAME}.out 2>&1 &


##### train psca-vae (different P_rand) - 2 GPUs

P_RAND=0.1
MODEL_NAME=pscavaetf-r${P_RAND}-test


CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29907 \
    Train1.py \
        -seed 1                      \
        -KLA_inc_beta 0.01           \
        -lr_WarmUpSteps 15000        \
        -use_cond2lat                \
        -model_type pscavaetf        \
        -start_epoch 1               \
        -num_epoch 20                \
        -property_list logP tPSA QED \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
        -use_scaffold                \
        -batch_size 64               \
        -randomize_prob ${P_RAND}    \
    >train-${MODEL_NAME}.out 2>&1 &


##### train psca-vae (molgpt) - 2 GPUs


# MODEL_NAME=pscavaetf1_molgpt-test


# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29903 \
#     Train1.py                        \
#         -seed 1                      \
#         -KLA_inc_beta 0.01           \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -model_type pscavaetf        \
#         -start_epoch 1               \
#         -num_epoch 50                \
#         -property_list logP tPSA SAS \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#         -use_scaffold                \
#         -batch_size 64               \
#         -randomize_prob 0.5          \
#     >train-${MODEL_NAME}.out 2>&1 &

