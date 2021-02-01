#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
INPUT=$4
OUTPUT=$5
echo $INPUT
echo $OUTPUT
echo ${@:4}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/inference.py $CONFIG $CHECKPOINT  $INPUT $OUTPUT --launcher pytorch ${@:6}