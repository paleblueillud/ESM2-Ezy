#!/usr/bin/env bash
set -euo pipefail

NPROC=${1:-1}

# For multi-node runs, launch with your scheduler's torchrun wrapper and set the appropriate env vars.
torchrun --nproc_per_node="${NPROC}" --nnodes=1 \
  scripts/generate_representations.py \
  -m esm2_t36_3B_UR50D \
  -ckpt ckpt/dnn_model_lastlayer1/best.pt \
  -p data/MCO_retrieval/new_positive_set.fasta \
  -n data/MCO_retrieval/new_negative_set.fasta \
  -s data/MCO_retrieval/reprs \
  -b 1 \
  --dtype float32
