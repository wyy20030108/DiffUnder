#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/abd.py \
 --cli_dataset "8busi" \
 --seed 2025 \
 --exp "ABD" \
 --device "cuda:$GPUID"