#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/agscore.py \
 --cli_dataset "20acdc" \
 --seed 2027 \
 --exp "AgScore" \
 --device "cuda:$GPUID"
