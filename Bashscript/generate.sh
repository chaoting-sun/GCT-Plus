#!/usr/bin/env bash

GPU_IDX=1

# Original Transformer

MODEL=tf
MODEL_TYPE=transformer
DECODE_TYPE=decode
SIMILARITY=1.00
EPOCH=3
DEMO=True

# MODEL=mlptf
# MODEL_TYPE=mlp_encoder
# DECODE_TYPE=mlp_decode
# SIMILARITY=0.70
# LOSS_FCN=kld
# EPOCH=3
# DEMO=True

# MLP-Encoder

# MODEL=mlptf
# MODEL_TYPE=mlp_encoder
# DECODE_TYPE=mlp_decode
# SIMILARITY=0.70
# LOSS_FCN=kld
# EPOCH=3
# DEMO=False

# ATT-Decoder 

# MODEL=atttf
# MODEL_VERSION=v5
# MODEL_TYPE=att_encoder
# DECODE_TYPE=att_decode
# SIMILARITY=0.80
# LOSS_FCN=kld
# EPOCH=3

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


# STD=1.0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     generate.py \
#         -variational \
#         -model_type mlp_encoder \
#     testing \
#         -epoch ${EPOCH} \
#         -decode_type mlp_decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${LOSS_FCN} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}_std${STD} \
#     >mlptf_generate_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}_${STD}.out 2>mlptf_generate_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}_${STD}.err &


if [ ${DEMO} == True ]
then
    CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
        generate.py \
            -similarity ${SIMILARITY} \
            -variational \
            -model_type ${MODEL_TYPE} \
        testing \
            -epoch ${EPOCH} \
            -decode_type ${DECODE_TYPE} \
            -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${LOSS_FCN} \
            -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY} \
            -demo
elif [ ${MODEL} == "tf" ]
then
    CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
        generate.py \
            -similarity ${SIMILARITY} \
            -variational \
            -model_type ${MODEL_TYPE} \
        testing \
            -decode_type ${DECODE_TYPE} \
            -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL}_sim1.00 \
        >${MODEL}_generate.out 2>${MODEL}_generate.err &
elif [ ${MODEL} == 'mlptf' ]
then
    # mlp_transformer - mlp_decoder
    CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
        generate.py \
            -similarity ${SIMILARITY} \
            -variational \
            -model_type ${MODEL_TYPE} \
        testing \
            -epoch ${EPOCH} \
            -decode_type ${DECODE_TYPE} \
            -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL}_train_stage2_sim${SIMILARITY}_${LOSS_FCN} \
            -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL}_sim${SIMILARITY}_epoch${EPOCH} \
        >${MODEL}_generate_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}.out 2>${MODEL}_generate_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}.err &
elif [ ${MODEL} == 'atttf' ]
then
    CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
        generate_att.py \
            -similarity ${SIMILARITY} \
            -variational \
            -model_type ${MODEL_TYPE} \
        testing \
            -epoch ${EPOCH} \
            -decode_type ${DECODE_TYPE} \
            -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/${MODEL}_train_stage2_sim${SIMILARITY}_${LOSS_FCN}_${MODEL_VERSION} \
            -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/${MODEL}_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH} \
        # >${MODEL}_generate_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}.out 2>${MODEL}_generate_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH}.err &
fi
