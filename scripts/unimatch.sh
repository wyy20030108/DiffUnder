#!/bin/bash
GPUID=1

CUDA_VISIBLE_DEVICES=$GPUID python comparsion/unimatch.py \
 --cli_dataset "20acdc" \
 --seed 2026 \
 --exp "unimatchv1-medical" \
 --device "cuda:$GPUID"