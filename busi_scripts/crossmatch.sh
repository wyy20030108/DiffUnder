#!/bin/bash
GPUID=6

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/crossmatch.py \
 --cli_dataset "32busi" \
 --seed 2027 \
 --exp "CrossMatch" \
 --device "cuda:$GPUID"