#!/usr/bin/env bash

CONFIG=/home/fu/workspace/RobustBEV/zoo/PolarFormer/projects/configs/polarformer/polarformer_r101.py
CHECKPOINT=/datasets/bev_models/polarformer_r101.pth
GPU_ID="0"

hsb='0.3'
shift='0.1'
scale='0.1'
motion='9'

iter='200'
nquery='100' # we use 2500 in the paper
deep='6'

mkdir ./results # only need to run once

CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 ./tools/eval.py $CONFIG $CHECKPOINT --eval bbox\
    --perturb-type optical --hue $hsb --saturation $hsb --bright $hsb\
    --max-evaluation $nquery --max-iteration $iter --max-deep $deep  \
    --save-result --save-name polar_hsb

CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 ./tools/eval.py $CONFIG $CHECKPOINT --eval bbox\
    --kernal-size $motion --perturb-type motion \
    --max-evaluation $nquery --max-iteration $iter --max-deep $deep  \
    --save-result --save-name polar_mb

CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 ./tools/eval.py $CONFIG $CHECKPOINT --eval bbox\
    --perturb-type geometry --shift $shift --scale $scale\
    --max-evaluation $nquery --max-iteration $iter --max-deep $deep  \
    --save-result --save-name polar_scsft