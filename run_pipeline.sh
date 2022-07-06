#!/usr/bin/env bash

set -x
set -e

CUDA_VISIBLE_DEVICES=$1 python generate_poison_transformer.py \
    cfg/singlesource_singletarget_1000class_finetune_deit_base/experiment_$2_base2.cfg

CUDA_VISIBLE_DEVICES=$1 python finetune_transformer.py \
    cfg/singlesource_singletarget_1000class_finetune_deit_base/experiment_$2_base2.cfg

