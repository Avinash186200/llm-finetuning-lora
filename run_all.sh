#!/usr/bin/env bash
# Reproduce every number in results/RESULTS.md. CPU, sequential.
set -euo pipefail
PY=${PY:-python}
$PY -m lorafin.cli make_data
for seed in 0 1 2; do
  $PY -m lorafin.cli train --set lora.r=8 lora.alpha=16 train.seed=$seed
  $PY -m lorafin.cli evaluate --method lora --adapter outputs/adapters/lora_r8_s$seed --label lora_r8_s$seed
done
for r in 4 16; do
  $PY -m lorafin.cli train --set lora.r=$r lora.alpha=$((2*r)) train.seed=0
  $PY -m lorafin.cli evaluate --method lora --adapter outputs/adapters/lora_r${r}_s0 --label lora_r${r}_s0
done
$PY -m lorafin.cli evaluate --method zero_shot
$PY -m lorafin.cli evaluate --method few_shot
$PY -m lorafin.cli evaluate --method zero_shot --retry
$PY -m lorafin.cli evaluate --method few_shot --retry
$PY -m lorafin.cli evaluate --method lora --retry --adapter outputs/adapters/lora_r8_s0 --label lora_r8_s0_retry
$PY -m lorafin.cli compare
$PY -m lorafin.cli export --adapter outputs/adapters/lora_r8_s0 --out outputs/merged_r8_s0
