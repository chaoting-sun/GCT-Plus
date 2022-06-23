#!/usr/bin/env bash 

python3 -u \
    main.py \
        -similarity 0.60 \
        -n_jobs 4 \
        -load_field \
        -data_name moses \
        -field_path ./molGCT \
        -load_scaler \
        -variational \


        