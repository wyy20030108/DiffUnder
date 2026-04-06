#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/abd.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "ABD" \
 --device "cuda:$GPUID"