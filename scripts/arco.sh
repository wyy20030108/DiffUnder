#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python comparsion/arco.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "ARCO-adamw" \
 --device "cuda:$GPUID"