#!/bin/bash
GPUID=2

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/cps.py \
 --cli_dataset "10acdc" \
 --seed 2025 \
 --exp "cps" \
 --device "cuda:$GPUID"