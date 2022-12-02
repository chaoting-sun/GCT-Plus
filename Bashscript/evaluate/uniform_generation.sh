#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + fromScratch
# > no src

# GPU_IDX=2

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# DECODE_ALGO="newbeam"
# EPOCH=30
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -n_jobs 4 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}/model_${EPOCH}.pt \
#     uniform-generation \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep${EPOCH} \
#         -uniform_generation \
#         -n_each_prop 5 \
#         -n_each_sampling 100 \
#     >>model-${MODEL_TYPE}_ep-${EPOCH}.out 2>&1 &


# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + fromScratch
# > no src

GPU_IDX=3

MODEL_TYPE=transformer
DECODE_TYPE=decode
DECODE_ALGO="newbeam"
# <<<<<<<<<<<<<<<<<<<<<<<<<

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -n_jobs 4 \
        -model_type ${MODEL_TYPE} \
        -use_model_path /fileserver-gamma/chaoting/ML/molGCT/molgct.pt \
    uniform-generation \
        -decode_algo ${DECODE_ALGO} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/molgct \
        -uniform_generation \
        -n_each_prop 5 \
        -n_each_sampling 100 \
    # >>model-molgct.out 2>&1 &