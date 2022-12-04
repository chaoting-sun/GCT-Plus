#!/usr/bin/env bash


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


################# varying conditions #################

# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + fromScratch
# > no src

# GPU_IDX=3
# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="beam"
# TOKLEN=40
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/train_${MODEL_TYPE}_fromScratch/ \
#         -use_epoch 7 \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_fromScratch_${CHOICE}_${DECODE_ALGO}_conds \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for conds \
#     >>model:${MODEL_TYPE}_fromScratch_algo:${DECODE_ALGO}_toklen:${TOKLEN}.out 2>&1 &


# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + trainDecoderOut
# > no src

# GPU_IDX=3

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="beam"
# TOKLEN=30
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/train_${MODEL_TYPE}_freezeDecoderOut/ \
#         -use_epoch 8 \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_freezeDecoderOut_${CHOICE}_${DECODE_ALGO}_conds \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for conds \
#     >>model:${MODEL_TYPE}_freezeDecoderOut_algo:${DECODE_ALGO}_toklen:${TOKLEN}.out 2>&1 &


############## varying z ##############

# # <<<<<<<<<<<<<<<<<<<<<<<<<
# # > model: transformer + fromScratch
# # > no src

# GPU_IDX=2

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="greedy"
# TOKLEN=30
# EPOCH=20
# # <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}/model_${EPOCH}.pt \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep${EPOCH}_ \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for conds \
#     # >>ContiCheck_model-${MODEL_TYPE}_ep-${EPOCH}.out 2>&1 &




# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: molgct
# > no src

# GPU_IDX=0

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="greedy"
# TOKLEN=30
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/molGCT/molgct.pt \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/molgct_ \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for conds \
    # >>model:${MODEL_TYPE}_freezeDecoderOut_algo:${DECODE_ALGO}_toklen:${TOKLEN}_zs.out 2>&1 &


# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + trainDecoderOut
# > no src

# GPU_IDX=2

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="greedy"
# TOKLEN=35
# <<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/train_${MODEL_TYPE}_freezeDecoderOut/ \
#         -use_epoch 19 \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_freezeDecoderOut_${CHOICE}_${DECODE_ALGO}_ep19_zs \
#         -continuity_check \
#         -properties 2.8421	58.1053	0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for z \
    # >>model:${MODEL_TYPE}_trainDecoderOut_algo:${DECODE_ALGO}_toklen:${TOKLEN}_zs.out 2>&1 &






# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
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
    # >>model:${MODEL_TYPE}_algo:${DECODE_ALGO}_toklen:${TOKLEN}.out 2>&1 &

#############################################################################

# <<<<<<<<<<<<<<<<<<<<<<<<<
# > model: molgct
# > no src

GPU_IDX=3
MODEL_TYPE=transformer
DECODE_TYPE=decode
SIMILARITY=1.00
EPOCH=3
CHOICE="continuity-check"
DECODE_ALGO="greedy"
TOKLEN=30
# <<<<<<<<<<<<<<<<<<<<<<<<<

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -similarity ${SIMILARITY} \
        -n_jobs 2 \
        -model_type ${MODEL_TYPE} \
        -use_model_path /fileserver-gamma/chaoting/ML/molGCT/molgct.pt \
    continuity-check \
        -decode_algo ${DECODE_ALGO} \
        -decode_type ${DECODE_TYPE} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/molgct \
        -continuity_check \
        -properties 2.8421	58.1053	0.8947 \
        -toklen ${TOKLEN} \
        -n_steps 50 \
        -n_samples 100 \
        -test_for z \
    # >>model:${MODEL_TYPE}_algo:${DECODE_ALGO}_toklen:${TOKLEN}.out 2>&1 &

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + aug
# > no src

# GPU_IDX=1

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="greedy"
# TOKLEN=30
# EPOCH=30
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -n_jobs 2 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}_ep25_aug-decoderout/model_${EPOCH}.pt \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep25_aug-decoderout_ep${EPOCH} \
#         -continuity_check \
#         -properties 2.8421 58.1053 0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for conds \
#     >>ContiCheck_model-${MODEL_TYPE}_ep-${EPOCH}.out 2>&1 &


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# > model: transformer + aug
# > no src

# GPU_IDX=1

# MODEL_TYPE=transformer
# DECODE_TYPE=decode
# CHOICE="continuity-check"
# DECODE_ALGO="greedy"
# TOKLEN=30
# EPOCH=26
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -n_jobs 4 \
#         -model_type ${MODEL_TYPE} \
#         -use_model_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL_TYPE}_ep25_aug-all/model_${EPOCH}.pt \
#     continuity-check \
#         -decode_algo ${DECODE_ALGO} \
#         -decode_type ${DECODE_TYPE} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL_TYPE}_ep25_aug-all_ep${EPOCH} \
#         -continuity_check \
#         -properties 2.8421 58.1053 0.8947 \
#         -toklen ${TOKLEN} \
#         -n_steps 50 \
#         -n_samples 100 \
#         -test_for z \
    # >>ContiCheck_model-${MODEL_TYPE}_ep-${EPOCH}.out 2>&1 &