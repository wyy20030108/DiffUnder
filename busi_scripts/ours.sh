#!/bin/bash
GPUID=2

CUDA_VISIBLE_DEVICES=$GPUID python c_busi/ot_arco_busi.py \
 --cli_dataset "8busi" \
 --seed 2027 \
 --exp "ARCO_OT" \
 --device "cuda:$GPUID"