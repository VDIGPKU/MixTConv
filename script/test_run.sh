#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ./../test_models.py somethingv1 \
    --weights=checkpoint/somethingv1_RGB_resnet50_ops_ms_group1douter_8_avg_segment8_e50/ckpt.best.pth.tar \
    --test_segments=8 --test_crops=1 \
    --batch_size=80 --full_res