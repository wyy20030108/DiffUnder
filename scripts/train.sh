#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python train.py \
 --cli_dataset "20acdc" \
 --seed 2026 \
 --exp "Ours" \
 --device "cuda:$GPUID"