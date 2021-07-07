#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python ../main.py somethingv1 RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --operations=ms_group1douter --n_div=8 --npb

