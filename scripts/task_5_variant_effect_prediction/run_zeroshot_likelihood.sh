#!/bin/bash

set -eu

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate veRL
cd ~/DART-Eval/src

export CUDA_VISIBLE_DEVICES=1,2,3
export DART_WORK_DIR=/data1/Mamba/Dataset/Genome/DART-Eval

MODEL=nucleotide_transformer
DART_WORK_DIR=/data1/Mamba/Dataset/Genome/DART-Eval
MODEL_NAME=nucleotide-transformer-v2-500m-multi-species
MODEL_PATH=/data1/Mamba/Model/Genome/Nucleotide-Transformer/nucleotide-transformer-v2-500m-multi-species
BATCH_SIZE=512


python -m dnalm_bench.task_2_5_single.experiments.task_5_variant_effect_prediction.zero_shot_likelihoods.$MODEL \
  --elements_tsv $DART_WORK_DIR/task_5_variant_effect_prediction/input_data/Afr.CaQTLS.tsv \
  --output_prefix Afr.CaQTLS \
  --genome_fa $DART_WORK_DIR/refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
  --batch_size $BATCH_SIZE \
  --model_name $MODEL_NAME \
  --model_path $MODEL_PATH
