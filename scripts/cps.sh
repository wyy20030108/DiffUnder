#!/bin/bash
GPUID=4

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/cps.py \
 --cli_dataset "5acdc" \
 --seed 2027 \
 --exp "cps" \
 --device "cuda:$GPUID"