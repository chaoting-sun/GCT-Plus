#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/python-plot/':$PYTHONPATH



# CTF
# SIMILARITY=1.00
# TOLERANCE=0.00

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     test.py                                         \
#         -model_type ctf \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
#     reconstruction                                  \
#         -reconstruction                             \
#         -model_name ctf                             \
#         -epoch_list 1 2 3 4 5 6 7 8 9 10                              \
#         -similarity ${SIMILARITY}                            \
#         -tolerance ${TOLERANCE}                             \
#     >>reconstruction_s${SIMILARITY}_t${TOLERANCE}.out 2>&1 &

        # -model_name ctf_aug-s${SIMILARITY}-t${TOLERANCE} \
        # -epoch_list 1 2 3 4 5 6 7 8 9 10            \


# GPU_IDX=0
# SIMILARITY=1.00
# TOLERANCE=0.00
# MODEL_NAME=ctf

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     test.py                                    \
#         -model_type cvaetf \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
#     reconstruction                                     \
#         -reconstruction                                \
#         -model_name cvaetf                             \
#         -epoch_list 10                                 \
#         -similarity 1.00                               \
#         -tolerance 0.00                                \
#         -pad_to_same_len                               \
    # >>rec_${MODEL_NAME}_s${SIMILARITY}_t${TOLERANCE}.out 2>&1 &


GPU_IDX=1
SIMILARITY=1.00
TOLERANCE=0.00

MODEL_NAME=cvaetf_pad
# MODEL_NAME=attenctf_pad-aug-s${SIMILARITY}-t${TOLERANCE}
# MODEL_NAME=ctf_aug-s${SIMILARITY}-t${TOLERANCE}
# MODEL_NAME=attenctf_pad-aug-s${SIMILARITY}-t${TOLERANCE}-half
# MODEL_NAME=attenctf_pad_dconds_gpt_aug-s0.50-t0.20
# MODEL_NAME=attenctf_pad_dconds_zeros_aug-s0.50-t0.20-half
# MODEL_NAME=attenctf_pad_aug-s0.50-t0.20_22


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    test.py                                                      \
        -model_type cvaetf                                       \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    reconstruction                                     \
        -reconstruction                                \
        -model_name ${MODEL_NAME}                      \
        -epoch_list 4                                  \
        -similarity ${SIMILARITY}                      \
        -tolerance ${TOLERANCE}                        \
        -pad_to_same_len                               \


        
    # >>rec_${MODEL_NAME}_s${SIMILARITY}_t${TOLERANCE}.out 2>&1 &

        # -pad_to_same_len                               \

        # -epoch_list 1 2                              \
