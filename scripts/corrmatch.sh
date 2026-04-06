#!/bin/bash
GPUID=0

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/CorrMatch.py \
 --cli_dataset "10acdc" \
 --seed 2025 \
 --exp "CorrMatch" \
 --device "cuda:$GPUID"