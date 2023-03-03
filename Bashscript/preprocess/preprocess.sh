#!/usr/bin/env bash

# moses

python3 -u \
    preprocess.py \
        -benchmark moses \
        -all_property_list 'logP' 'tPSA' 'QED' 'SAS' \
        -similarity_threshold 0.70 \
        -n_jobs 16 \
    # >/dev/null 2>&1 &

# chembl_02

# python3 -u \
#     preprocess.py \
#         -benchmark chembl_02 \
#         -all_property_list logP \
#         -n_jobs 32 \