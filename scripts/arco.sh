#!/bin/bash
GPUID=6

CUDA_VISIBLE_DEVICES=$GPUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python comparsion/arco.py \
 --cli_dataset "20acdc" \
 --seed 2025 \
 --exp "ARCO-adamw" \
 --device "cuda:$GPUID"