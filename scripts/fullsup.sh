#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/fullsup.py \
 --cli_dataset "20acdc" \
 --seed 2027 \
 --exp "FullySupervised" \
 --device "cuda:$GPUID"