#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + fromScratch
# > no src

# GPU_IDX=3

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# DECODE_ALGO="greedy"

# INIT_EPOCH=25
# USE_EPOCH=30
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -n_jobs 4 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}_ep${INIT_EPOCH}_aug-decoderout/model_${USE_EPOCH}.pt \
#     uniform-generation \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep${INIT_EPOCH}_aug-decoderout_ep${USE_EPOCH} \
#         -uniform_generation \
#         -n_each_prop 5 \
#         -n_each_sampling 100 \
#     >>${MODEL_TYPE}_ep${INIT_EPOCH}_aug-decoderout_ep${USE_EPOCH}.out 2>&1 &

# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + fromScratch
# > no src

GPU_IDX=3
MODEL_NAME=transformer
MODEL_NUM=3
# <<<<<<<<<<<<<<<<<<<<<<<<<

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
        -model_type transformer                     \
    uniform-generation                              \
        -uniform_generation                         \
        -epoch_list 16 17 18 19 20 21 22 23 24 25   \
        -model_name ${MODEL_NAME}${MODEL_NUM}       \
    # >>ContiCheck_model-${MODEL_NAME}{MODEL_NUM}.out 2>&1 &


# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + fromScratch
# > no src

# GPU_IDX=0

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# DECODE_ALGO="greedy"
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -n_jobs 4 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/molGCT/molgct.pt \
#     uniform-generation \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/molgct \
#         -uniform_generation \
#         -n_each_prop 5 \
#         -n_each_sampling 100 \
    # >>model-molgct.out 2>&1 &