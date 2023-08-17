#!/usr/bin/env bash


##### Our settings


# SCAFFOLD_SOURCE=train # test_scaffolds


# CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     psca-sampling \
#         -property_list logP tPSA QED \
#         -model_type pscavaetf \
#         -model_name pscavaetf1.pt \
#         -model_folder ./Weights/pscavaetf/ \
#         -save_folder ./Data/inference/sca-sampling/pscavaetf1/${SCAFFOLD_SOURCE}/ \
#         -scaffold_source ${SCAFFOLD_SOURCE} \
#         -decode_algo multinomial                                         \
#         -n_samples 10000 \
    # >>psca_sampling.out 2>&1 &


##### Our settings 


SCAFFOLD_SOURCE=test_scaffolds
P_RANDOMIZE=0.1


CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -use_cond2lat \
        -use_scaffold \
    psca-sampling \
        -property_list logP tPSA QED \
        -model_type pscavaetf \
        -model_name pscavaetf1_r${P_RANDOMIZE}.pt \
        -model_folder ./Weights/pscavaetf/ \
        -save_folder ./Data/inference/sca-sampling/pscavaetf1_r${P_RANDOMIZE}/${SCAFFOLD_SOURCE}/ \
        -scaffold_source ${SCAFFOLD_SOURCE} \
        -decode_algo multinomial                                         \
        -n_samples 10000 \
    >>psca_sampling.out 2>&1 &


##### MolGPT settings


# SCAFFOLD_SOURCE=molgpt


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     psca-sampling \
#         -property_list logP tPSA SAS \
#         -model_type pscavaetf \
#         -model_name pscavaetf1_molgpt.pt \
#         -model_folder ./Weights/pscavaetf/ \
#         -save_folder ./Data/inference/sca-sampling/pscavaetf1/${SCAFFOLD_SOURCE}/ \
#         -scaffold_source ${SCAFFOLD_SOURCE} \
#         -decode_algo multinomial \
#         -n_samples 10000 \
#     >>psca_sampling.out 2>&1 &
