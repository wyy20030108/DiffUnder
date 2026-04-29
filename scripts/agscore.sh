#!/bin/bash
GPUID=6

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/agscore.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "AgScore" \
 --device "cuda:$GPUID"
