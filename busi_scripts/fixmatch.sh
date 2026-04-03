#!/bin/bash
GPUID=6

CUDA_VISIBLE_DEVICES=$GPUID python fixmatch_busi.py \
 --cli_dataset "10acdc" \
 --seed 2027 \
 --exp "FixMatch" \
 --device "cuda:$GPUID"