#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/fullsup.py \
 --cli_dataset "16busi" \
 --seed 2025 \
 --exp "FullSup" \
 --device "cuda:$GPUID"