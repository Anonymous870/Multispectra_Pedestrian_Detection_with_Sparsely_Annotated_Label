#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path Model_weight --result-dir save_path #add "--vis"  #if you want to visualize the result