#!/bin/bash
GPUID=2

CUDA_VISIBLE_DEVICES=$GPUID python JEPA_OT.py \
 --cli_dataset "5acdc" \
 --seed 2026 \
 --exp "Ours" \
 --device "cuda:$GPUID"