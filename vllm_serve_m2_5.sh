#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SAFETENSORS_FAST_GPU=1

# good hygiene for multi-proc startup
unset CUDA_MODULE_LOADING
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_FLASHINFER_MOE_FP8=0

vllm serve MiniMaxAI/MiniMax-M2.5 \
  --trust-remote-code \
  --enable-expert-parallel --tensor-parallel-size 8 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --enable-auto-tool-choice \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --host 0.0.0.0 \
  --port 8000 \
  --seed 0
