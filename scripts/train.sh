#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python train.py \
 --cli_dataset "10acdc" \
 --seed 2025 \
 --exp "Ours" \
 --device "cuda:$GPUID"