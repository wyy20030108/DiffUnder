#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/abd.py \
 --cli_dataset "32busi" \
 --seed 2025 \
 --exp "ABD" \
 --device "cuda:$GPUID"