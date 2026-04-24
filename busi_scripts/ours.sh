#!/bin/bash
GPUID=3

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/ot_arco_v1.1.py \
 --cli_dataset "8busi" \
 --seed 2026 \
 --exp "ARCO-UOT" \
 --device "cuda:$GPUID"