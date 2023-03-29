#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

# GPU_IDX=0
# MODEL_TYPE=cvaetf
# MODEL_NAME=cvaetf
# BENCHMARK=chembl_02

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                    \
#         -model_type ${MODEL_TYPE} \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset \
#     src-generation-mmps \
#         -src_generation_mmps \
#         -benchmark ${BENCHMARK} \
#         -property_list logP \
#         -decode_algo multinomial \
#         -model_name ${MODEL_NAME} \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -data_name test \
#         -n_steps 2 \
#         -epoch_list 20 \


GPU_IDX=2
SIMILARITY=1.00

MODEL_TYPE=attencvaetf
MODEL_NAME=${MODEL_TYPE}-mconds-s0.70
EPOCH=5

# MODEL_TYPE=attencvaetf
# MODEL_NAME=${MODEL_TYPE}-dconds-s0.70
# EPOCH=2

# MODEL_TYPE=attencvaetf
# MODEL_NAME=${MODEL_TYPE}-z-s0.70
# EPOCH=2

# MODEL_TYPE=cvaetf
# MODEL_NAME=${MODEL_TYPE}-s1.00
# EPOCH=25

BENCHMARK=moses

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -model_type ${MODEL_TYPE} \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset \
    src-generation-mmps \
        -src_generation_mmps \
        -benchmark ${BENCHMARK} \
        -property_list logP tPSA QED \
        -decode_algo multinomial \
        -model_name ${MODEL_NAME} \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -data_name test_scaffolds-s${SIMILARITY} \
        -n_steps 1 \
        -epoch_list ${EPOCH} \
    # >src_gen-${MODEL_NAME}.out 2>&1 &

    # >/dev/null 2>&1 &

# GPU_IDX=2
# MODEL_TYPE=attencvaetf
# BENCHMARK=moses
# SIMILARITY=0.70

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                    \
#         -model_type ${MODEL_TYPE} \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
#     src-generation-mmps \
#         -src_generation_mmps \
#         -benchmark ${BENCHMARK} \
#         -property_list logP tPSA QED \
#         -decode_algo multinomial \
#         -model_name ${MODEL_TYPE} \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -data_name test-s${SIMILARITY} \
#         -n_steps 1 \
#         -epoch_list 2 \

#     # >/dev/null 2>&1 &

#     #     -model_name ${MODEL_TYPE}-s${SIMILARITY} \
