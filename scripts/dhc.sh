#!/bin/bash
GPUID=3

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/dhc.py \
 --cli_dataset "20acdc" \
 --seed 2025 \
 --exp "DHC" \
 --device "cuda:$GPUID"