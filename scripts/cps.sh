#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/cps.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "cps" \
 --device "cuda:$GPUID"