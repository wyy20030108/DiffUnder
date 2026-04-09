#!/bin/bash
GPUID=0

CUDA_VISIBLE_DEVICES=$GPUID python fixmatch_busi.py \
 --cli_dataset "32busi" \
 --seed 2027 \
 --exp "FixMatch_new" \
 --device "cuda:$GPUID"