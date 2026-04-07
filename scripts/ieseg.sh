#!/bin/bash
GPUID=4
t=100

CUDA_VISIBLE_DEVICES=$GPUID python IESeg.py \
 --cli_dataset "20acdc" \
 --seed 2027 \
 --exp "IESeg_t${t}_V1" \
 --stage1_epochs ${t} \
 --device "cuda:$GPUID"
