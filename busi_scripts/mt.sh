#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/mt.py \
 --cli_dataset "32busi" \
 --seed 2027 \
 --exp "MT" \
 --device "cuda:$GPUID"