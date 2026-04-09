#!/bin/bash
GPUID=6

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/cps.py \
 --cli_dataset "8busi" \
 --seed 2025 \
 --exp "cps" \
 --device "cuda:$GPUID"