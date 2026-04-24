#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/arco-uot.py \
 --cli_dataset "5acdc" \
 --seed 2025 \
 --exp "ARCO-UOT-AdamW" \
 --device "cuda:$GPUID"