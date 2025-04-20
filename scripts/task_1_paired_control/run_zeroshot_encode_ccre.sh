#!/bin/bash

set -eu

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate veRL
cd ~/DART-Eval/src

export CUDA_VISIBLE_DEVICES=1,2,3
export DART_WORK_DIR=/data1/Mamba/Dataset/Genome/DART-Eval

DART_WORK_DIR=/data1/Mamba/Dataset/Genome/DART-Eval
BATCH_SIZE=2048

# MODEL=nucleotide_transformer
# MODEL_NAME=nucleotide-transformer-v2-500m-multi-species
# MODEL_PATH=/data1/Mamba/Model/Genome/Nucleotide-Transformer/nucleotide-transformer-v2-500m-multi-species

# MODEL=geneator
# MODEL_NAME=GENERator-eukaryote-1.2b-base
# MODEL_PATH=/data1/Mamba/Model/Genome/GENERator/GENERator-eukaryote-1.2b-base

# MODEL=hyenadna
# MODEL_NAME=hyenadna-large-1m-seqlen-hf
# MODEL_PATH=/data1/Mamba/Model/Genome/HyenaDNA/hyenadna-large-1m-seqlen-hf

# MODEL=gena_lm
# MODEL_NAME=gena-lm-bert-base-t2t
# MODEL_PATH=/data1/Mamba/Model/Genome/GENA-LM/gena-lm-bert-base-t2t

MODEL=dnabert2
MODEL_NAME=DNABERT-2-117M
MODEL_PATH=/data1/Mamba/Model/Genome/DNABERT-2/DNABERT-2-117M

python -m dnalm_bench.task_1_paired_control.zero_shot.encode_ccre.$MODEL \
  --work_dir $DART_WORK_DIR \
  --batch_size $BATCH_SIZE \
  --model_name $MODEL_NAME \
  --model_path $MODEL_PATH
