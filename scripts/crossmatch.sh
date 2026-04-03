#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/crossmatch.py \
 --cli_dataset "20acdc" \
 --seed 2025 \
 --exp "CrossMatch" \
 --device "cuda:$GPUID"