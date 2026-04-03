#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/cgs.py \
 --cli_dataset "5acdc" \
 --seed 2025 \
 --exp "CGS" \
 --device "cuda:$GPUID"