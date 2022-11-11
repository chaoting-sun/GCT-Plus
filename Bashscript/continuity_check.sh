#!/usr/bin/env bash

GPU_IDX=3

MODEL_TYPE=transformer
DECODE_TYPE=decode
SIMILARITY=1.00
EPOCH=3
CHOICE="continuity-check"
DECODE_ALGO="multinomial"
TOKLEN=40


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -similarity ${SIMILARITY} \
#         -n_jobs 2 \
#         -variational \
#         -model_type ${MODEL_TYPE} \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_${CHOICE}_${DECODE_ALGO}_z \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for z \
#     >>_model:${MODEL_TYPE}_toklen:${TOKLEN}_gpu:${GPU_IDX}.out 2>&1 &

########## no src & varying conds ##########

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -variational \
        -model_type ${MODEL_TYPE} \
    continuity-check \
        -decode_algo ${DECODE_ALGO} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_${CHOICE}_${DECODE_ALGO}_conds \
        -continuity_check \
        -properties 2.8421	58.1053	0.8947 \
        -toklen ${TOKLEN} \
        -n_steps 50 \
        -n_samples 100 \
        -test_for conds \
    >>model:${MODEL_TYPE}_algo:${DECODE_ALGO}_toklen:${TOKLEN}.out 2>&1 &

########## no src & varying conds ##########

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -similarity ${SIMILARITY} \
#         -n_jobs 2 \
#         -variational \
#         -model_type ${MODEL_TYPE} \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_${CHOICE}_${DECODE_ALGO}_conds \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#     >>model:${MODEL_TYPE}_algo:${DECODE_ALGO}_toklen:${TOKLEN}.out 2>&1 &