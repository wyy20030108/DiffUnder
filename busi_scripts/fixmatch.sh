#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID python fixmatch_busi.py \
 --cli_dataset "20acdc" \
 --seed 2027 \
 --exp "FixMatch" \
 --device "cuda:$GPUID"