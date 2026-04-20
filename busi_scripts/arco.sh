#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python c_busi/arco.py \
 --cli_dataset "8busi" \
 --seed 2026 \
 --exp "ARCO" \
 --device "cuda:$GPUID"