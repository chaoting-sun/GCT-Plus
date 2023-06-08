#!/usr/bin/env bash

# moses

python3 \
    preprocess.py \
        -benchmark moses \
        -prepared_properties logP tPSA SAS \
        -n_jobs 8 \
    > preprocess.out 2>&1 & 
    # >/dev/null 2>&1 &

# chembl_02

# python3 -u \
#     preprocess.py \
#         -benchmark chembl_02 \
#         -all_property_list logP \
#         -n_jobs 32 \