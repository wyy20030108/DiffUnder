#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/FixMatch.py \
 --cli_dataset "5acdc" \
 --seed 2025 \
 --exp "FixMatch" \
 --device "cuda:$GPUID"