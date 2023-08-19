#!/usr/bin/env bash

# moses

python3 \
    preprocess.py \
        -benchmark moses \
        -prepared_properties logP tPSA SAS \
        -n_jobs 8 \
    > preprocess.out 2>&1 & 
    # >/dev/null 2>&1 &
