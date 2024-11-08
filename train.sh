#!/bin/bash

cd src

CUDA_VISIBLE_DEVICES=0 python train_eval.py

exit 0