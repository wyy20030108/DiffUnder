#!/bin/bash
GPUID=0

CUDA_VISIBLE_DEVICES=$GPUID python train.py \
 --cli_dataset "20acdc" \
 --seed 2025 \
 --exp "Ours-lr0.1" \
 --device "cuda:$GPUID"