#!/bin/bash
GPUID=3

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/cgs.py \
 --cli_dataset "20acdc" \
 --seed 2025 \
 --exp "CGS" \
 --device "cuda:$GPUID"