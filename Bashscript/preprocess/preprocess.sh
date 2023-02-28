#!/usr/bin/env bash

# export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity':$PYTHONPATH

# augment data by molecular similarity and properties

TOLERANCE=1.00
SIMILARITY=0.40

python3 -u \
    preprocess.py \
        -tolerance ${TOLERANCE}  \
        -similarity ${SIMILARITY} \
        -n_jobs 4 \
    # >>preprocess_t${TOLERANCE}_s${SIMILARITY}.out 2>&1 &
    