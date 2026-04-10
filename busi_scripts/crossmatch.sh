#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/crossmatch.py \
 --cli_dataset "16busi" \
 --seed 2025 \
 --exp "CrossMatch" \
 --device "cuda:$GPUID"