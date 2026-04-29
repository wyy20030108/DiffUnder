#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/arco-uot.py \
 --cli_dataset "20acdc" \
 --seed 2026 \
 --exp "ARCO-UOT-AdamW-2" \
 --device "cuda:$GPUID"