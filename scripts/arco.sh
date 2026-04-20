#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python comparsion/arco.py \
 --cli_dataset "20acdc" \
 --seed 2027 \
 --exp "ARCO" \
 --device "cuda:$GPUID"