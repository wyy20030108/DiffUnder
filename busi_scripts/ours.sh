#!/bin/bash
GPUID=3

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/ours.py \
 --cli_dataset "8busi" \
 --seed 2027 \
 --exp "FSGA" \
 --device "cuda:$GPUID"