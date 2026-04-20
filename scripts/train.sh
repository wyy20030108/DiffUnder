#!/bin/bash
GPUID=2

CUDA_VISIBLE_DEVICES=$GPUID python ARCO_OT.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "ARCO_OT" \
 --device "cuda:$GPUID"