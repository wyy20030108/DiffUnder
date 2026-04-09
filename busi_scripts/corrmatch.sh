#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/corrmatch.py \
 --cli_dataset "8busi" \
 --seed 2027 \
 --exp "CorrMatch" \
 --device "cuda:$GPUID"