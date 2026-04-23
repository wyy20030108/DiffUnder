#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python c_busi/arco.py \
 --cli_dataset "16busi" \
 --seed 2025 \
 --exp "ARCO-adamw" \
 --device "cuda:$GPUID"