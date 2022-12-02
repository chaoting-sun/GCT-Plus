#!/usr/bin/env bash

GPU_IDX=3

MODEL_TYPE='att_encoder'
ENCODE_TYPE=encode_att_sample
DECODE_TYPE=decode
SIMILARITY=1.00
EPOCH=2
CHOICE="self-attention"
DECODE_ALGO="multinomial"
TOKLEN=45


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -variational \
        -model_type ${MODEL_TYPE} \
        -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/atttf_train_stage2_sim0.80_kld_v5/ \
        -use_epoch ${EPOCH} \
    ${CHOICE} \
        -decode_algo ${DECODE_ALGO} \
        -encode_type ${ENCODE_TYPE} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_${CHOICE}_${DECODE_ALGO} \
        -smiles 'CNC(=O)c1cccc(NCC(=O)Nc2cccc(C(=O)NC)c2)c1' \
        -toklen ${TOKLEN} \
        -n_samples 100 \
        -target_props 2.8421 58.1053 0.8947
        
        # 1.5 100 0.65
        
    # >>model:${MODEL_TYPE}_toklen:${TOKLEN}_gpu:${GPU_IDX}.out 2>&1 &

