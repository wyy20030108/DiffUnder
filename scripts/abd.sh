#!/bin/bash
GPUID=6

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/abd.py \
 --cli_dataset "5acdc" \
 --seed 2025 \
 --exp "ABD" \
 --device "cuda:$GPUID"