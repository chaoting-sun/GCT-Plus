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

GPU_IDX=1

MODEL_TYPE=transformer
DECODE_TYPE=decode
DECODE_ALGO="greedy"
EPOCH=25
# <<<<<<<<<<<<<<<<<<<<<<<<<

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py \
        -n_jobs 4 \
        -model_type ${MODEL_TYPE} \
        -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}/model_${EPOCH}.pt \
    uniform-generation \
        -decode_algo ${DECODE_ALGO} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep${EPOCH} \
        -uniform_generation \
        -n_each_prop 5 \
        -n_each_sampling 100 \
    >>model-${MODEL_TYPE}_ep-${EPOCH}.out 2>&1 &


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