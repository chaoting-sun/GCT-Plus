#!/usr/bin/env bash


##### Train (multiple GPUs)


MODEL_NAME=pscavaetf1


CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --master_port 29990 \
    train1.py \
        -seed 1                      \
        -model_type pscavaetf        \
        -lr_WarmUpSteps 15000        \
        -use_cond2lat                \
        -use_scaffold                \
        -start_epoch 1               \
        -num_epoch 50                \
        -batch_size 64               \
        -property_list logP tPSA QED \
        -model_folder ./Experiment/${MODEL_NAME} \
    # >train-${MODEL_NAME}.out 2>&1 &


##### Train with different P_rand (multiple GPUs)


# P_RAND=0.1
# MODEL_NAME=pscavaetf1-r${P_RAND}


# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29907 \
#     train1.py \
#         -seed 1                      \
#         -model_type pscavaetf        \
#         -KLA_inc_beta 0.01           \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -use_scaffold                \
#         -start_epoch 1               \
#         -num_epoch 20                \
#         -batch_size 64               \
#         -property_list logP tPSA QED \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#         -randomize_prob ${P_RAND}    \
#     >train-${MODEL_NAME}.out 2>&1 &


##### Train to compare with MolGPT (multiple GPUs)


# MODEL_NAME=pscavaetf1_molgpt


# CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29903 \
#     train1.py                        \
#         -seed 1                      \
#         -model_type pscavaetf        \
#         -KLA_inc_beta 0.01           \
#         -lr_WarmUpSteps 15000        \
#         -use_cond2lat                \
#         -use_scaffold                \
#         -start_epoch 1               \
#         -num_epoch 50                \
#         -batch_size 64               \
#         -property_list logP tPSA SAS \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL_NAME} \
#         -randomize_prob 0.5          \
#     >train-${MODEL_NAME}.out 2>&1 &


