#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/dhc.py \
 --cli_dataset "10acdc" \
 --seed 2025 \
 --exp "DHC" \
 --device "cuda:$GPUID"