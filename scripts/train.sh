#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/arco-uot.py \
 --cli_dataset "5acdc" \
 --seed 2027 \
 --exp "ARCO-UOT-AdamW" \
 --device "cuda:$GPUID"