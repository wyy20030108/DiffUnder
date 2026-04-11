#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/ours.py \
 --cli_dataset "32busi" \
 --seed 2027 \
 --exp "FSGA" \
 --device "cuda:$GPUID"