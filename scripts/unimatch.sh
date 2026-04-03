#!/bin/bash
GPUID=0

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/unimatch.py \
 --cli_dataset "20acdc" \
 --seed 2026 \
 --exp "unimatchv1-medical-0" \
 --device "cuda:$GPUID"