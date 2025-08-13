#!/bin/bash
# Quick training script
echo "Starting Digital Watermarking Training"
python train.py --epochs 50 --batch_size 32 --use_cifar --device cuda
