#!/bin/bash
GPUID=0

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/abd.py \
 --cli_dataset "32busi" \
 --seed 2026 \
 --exp "ABD" \
 --device "cuda:$GPUID"