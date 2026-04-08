#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/agscore.py \
 --cli_dataset "5acdc" \
 --seed 2027 \
 --exp "AgScore" \
 --device "cuda:$GPUID"
