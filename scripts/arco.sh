#!/bin/bash
GPUID=2

CUDA_VISIBLE_DEVICES=$GPUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python comparsion/arco.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "ARCO" \
 --device "cuda:$GPUID"