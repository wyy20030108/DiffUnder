#!/bin/bash
GPUID=3

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/jepa_ours.py \
 --cli_dataset "8busi" \
 --seed 2027 \
 --exp "OT-JEPA" \
 --device "cuda:$GPUID"