#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


GPU_IDX=2
BENCHMARK=moses


##### our trained model


MODEL=pscavaetf1
# MODEL=pscavaetf1_r0.5
EPOCH=20


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py                                    \
        -use_cond2lat                               \
    psca-sampling                                   \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -decode_algo multinomial                                         \
        -model_type scacvaetfv3                                        \
        -model_name model_${EPOCH}.pt               \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/${MODEL}-${EPOCH} \
        -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling \
        -property_list logP tPSA QED                                     \
        -n_samples 10000 \
        -scaffold_source test_scaffolds                                 \
    >>psca_sampling-${MODEL}-${GPU_IDX}.out 2>&1 &


##### our trained model (molgpt)


# GPU_IDX=0
# MODEL=pscavaetf1_molgpt


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                    \
#         -use_cond2lat                               \
#     psca-sampling                                   \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -decode_algo multinomial                                         \
#         -model_type scacvaetfv3                                        \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/${MODEL}-${EPOCH} \
#         -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling \
#         -property_list logP tPSA SAS                                     \
#         -n_samples 10000 \
#         -scaffold_source molgpt                                 \
#     # >>psca_sampling-${MODEL}-${GPU_IDX}.out 2>&1 &
