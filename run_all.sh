#!/usr/bin/env bash
# Full pipeline: data -> two LoRA adapters -> evaluate five systems -> report.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}

$PY -m lora_ft.data
$PY -m lora_ft.train_lora --recipe qa --epochs 3
$PY -m lora_ft.train_lora --recipe qa_passages --epochs 3
$PY -m lora_ft.evaluate --split test
$PY -m lora_ft.report --split test
