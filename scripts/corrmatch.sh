#!/bin/bash
GPUID=5

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/CorrMatch.py \
 --cli_dataset "10acdc" \
 --seed 2027 \
 --exp "CorrMatch" \
 --device "cuda:$GPUID"