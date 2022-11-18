#!/usr/bin/env bash

# export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity':$PYTHONPATH

# # nohup python3 -u \
# python3 -u \
#     main_preprocess.py \
#         -similarity 0.7 \
#         -n_jobs 8 \
#         -load_field \
#         -field_path ./molGCT \
#         -load_scaler \
#         -variational \
#         # >preprocess.out 2>preprocess.err&


# python3 -u \
#     main_preprocess.py \
#         -similarity 0.7 \
#         -n_jobs 8 \
#         -load_field \
#         -field_path ./molGCT \
#         -load_scaler \
#         -model_type 'mlp' \
#         -variational \
#         # >preprocess.out 2>preprocess.err&

# MODEL_TYPE='mlp_encoder'
# SIMILARITY=0.9

# python3 -u \
#     main_preprocess.py \
#         -similarity ${SIMILARITY} \
#         -n_jobs 4 \
#         -load_field \
#         -field_path ./molGCT \
#         -load_scaler \
#         -model_type ${MODEL_TYPE} \
#         -variational \
#     # >preprocess_${SIMILARITY}.out 2>preprocess_${SIMILARITY}.err&


python3 -u \
    preprocess.py \
        -tolerance 0.01 \
        -n_jobs 4 \
    # >preprocess_${SIMILARITY}.out 2>preprocess_${SIMILARITY}.err&