#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/fullsup.py \
 --cli_dataset "10acdc" \
 --seed 2025 \
 --exp "Fully Supervised" \
 --device "cuda:$GPUID"