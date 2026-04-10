#!/bin/bash
GPUID=3

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/unimatch.py \
 --cli_dataset "8busi" \
 --seed 2027 \
 --exp "UniMatch" \
 --device "cuda:$GPUID"