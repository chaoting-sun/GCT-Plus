#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer_ep25_aug-decoderout_ep${EPOCH}

GPU_IDX=1
MODEL_TYPE=transformer
DECODE_TYPE=decode
CHOICE="continuity-check"
DECODE_ALGO="greedy"
TOKLEN=30
EPOCH=29
OPTIM=adagrad
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py \
        -n_jobs 2 \
        -model_type ${MODEL_TYPE} \
        -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}_ep25_aug-decoderout_${OPTIM}/model_${EPOCH}.pt \
    continuity-check \
        -decode_algo ${DECODE_ALGO} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep25_aug-decoderout-${OPTIM}_ep${EPOCH} \
        -continuity_check \
        -properties 2.8421 58.1053 0.8947 \
        -toklen ${TOKLEN} \
        -n_steps 50 \
        -n_samples 100 \
        -test_for z \
    >>${MODEL_TYPE}_ep25_aug-decoderout-${OPTIM}_ep${EPOCH}.out 2>&1 &