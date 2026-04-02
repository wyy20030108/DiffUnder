#!/bin/bash
GPUID=7

CUDA_VISIBLE_DEVICES=$GPUID python train.py \
 --cli_dataset "5acdc" \
 --seed 2027 \
 --exp "Ours" \
 --device "cuda:$GPUID"