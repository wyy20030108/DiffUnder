#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/unimatch.py \
 --cli_dataset "16busi" \
 --seed 2025 \
 --exp "UniMatch" \
 --device "cuda:$GPUID"