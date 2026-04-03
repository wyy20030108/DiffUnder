#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/cgs.py \
 --cli_dataset "20acdc" \
 --seed 2026 \
 --exp "CGS" \
 --device "cuda:$GPUID"