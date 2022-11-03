#!/usr/bin/env bash

GPU_IDX=3

MODEL_TYPE=transformer
DECODE_TYPE=decode
SIMILARITY=1.00
EPOCH=3
CHOICE="continuity-check"
DECODE_ALGO="beam"
TOKLEN=45


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -variational \
        -model_type ${MODEL_TYPE} \
    continuity-check \
        -decode_algo ${DECODE_ALGO} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_${CHOICE}_${DECODE_ALGO} \
        -continuity_check \
        -properties 2.8421	58.1053	0.8947 \
        -toklen ${TOKLEN} \
        -n_steps 50 \
        -n_samples 100 \
    # >>model:${MODEL_TYPE}_toklen:${TOKLEN}_gpu:${GPU_IDX}.out 2>&1 &

