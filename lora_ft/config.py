"""Shared settings: model names, prompts, and file locations."""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "adapters"
RESULTS_DIR = ROOT / "results"

DATASET = "rag-datasets/rag-mini-wikipedia"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

SEED = 42
# Fractions of the 918 questions; whatever is left over becomes the test set.
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1

ABSTAIN = "I don't know"

CLOSED_BOOK_SYSTEM = (
    "You are a concise question-answering assistant. Answer with a short phrase, "
    "or with 'Yes' or 'No' for yes/no questions. "
    f"If you do not know the answer, reply exactly: {ABSTAIN}."
)

RAG_SYSTEM = (
    "You are a concise question-answering assistant. Answer using only the context "
    "provided. Answer with a short phrase, or with 'Yes' or 'No' for yes/no questions. "
    f"If the context does not contain the answer, reply exactly: {ABSTAIN}."
)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype() -> torch.dtype:
    # bf16 halves memory and works on recent CUDA GPUs and Apple Silicon.
    return torch.bfloat16 if device() in ("cuda", "mps") else torch.float32
