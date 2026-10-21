# LoRA-FineTuning

Fine-tune a small open model with **LoRA** (HuggingFace PEFT), then measure it
against a **RAG** pipeline on held-out questions, reporting **accuracy** and
**hallucination rate**.

The question this project answers: *if you want a model to answer questions
about a body of documents, should you bake the documents into the weights with
fine-tuning, or retrieve them at question time?*

## Setup at a glance

| Piece | Choice |
|---|---|
| Base model | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Fine-tuning | LoRA, rank 16, alpha 32, on all attention + MLP projections (~8.8M trainable params, 1.8% of the model) |
| Dataset | [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia): 918 human-written questions, ~3,000 Wikipedia passages |
| Split | 643 train / 92 val / 183 **held-out test**, stratified by yes/no vs factoid |
| Retriever | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings, cosine top-3 |
| Hardware | Apple M2 Pro (MPS), 16 GB |

## The five systems

| System | Knowledge comes from |
|---|---|
| **Base model** | whatever it learned in pre-training (closed book) |
| **LoRA: Q&A only** | training Q&A pairs. These teach answer *style*, but the facts behind the test questions are never shown |
| **LoRA: Q&A + passages** | training Q&A pairs **plus every corpus passage** as plain text, which puts the same knowledge RAG retrieves into the weights |
| **RAG** | base model + top-3 retrieved passages in the prompt |
| **LoRA + RAG** | the Q&A + passages adapter, also given retrieved passages |

Comparing the two LoRA systems separates *learning a format* from *learning facts*.

## Metrics

Each answer gets exactly one label (`lora_ft/metrics.py`):

- **correct**: yes/no questions must match on the first word; factoid answers
  must contain the gold answer (whole words, after normalization), be contained
  in it, or have token-F1 ≥ 0.5 with it.
- **abstained**: the model said it doesn't know. All systems are told to reply
  *"I don't know"* rather than guess.
- **hallucinated**: the model gave an answer and it was wrong.

**Accuracy** = correct / total. **Hallucination rate** = hallucinated / total.
They don't sum to 100%, because an honest "I don't know" is neither.

## Results

> ⏳ Training is still running. This section will be filled in with the numbers
> from `results/` when it finishes.

## Run it yourself

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PY=python ./run_all.sh
```

Or step by step:

```bash
python -m lora_ft.data                                # download, clean, split
python -m lora_ft.train_lora --recipe qa              # ~8 min on an M2 Pro
python -m lora_ft.train_lora --recipe qa_passages     # longer: 3,658 examples x 3 epochs
python -m lora_ft.evaluate --split test               # all five systems
python -m lora_ft.report --split test                 # table + chart in results/
pytest tests
```

Runs on CUDA, Apple Silicon (MPS) or CPU. To try a bigger model, change
`BASE_MODEL` in `lora_ft/config.py`.

## Project layout

```
lora_ft/
  config.py      model names, prompts, paths, device selection
  data.py        download, fix encoding, stratified split -> data/*.jsonl
  train_lora.py  LoRA fine-tuning with PEFT + HF Trainer (prompt tokens masked from loss)
  rag.py         dense retriever over the passage corpus
  evaluate.py    answer the held-out set with each system, grade, save predictions
  metrics.py     correct / abstained / hallucinated grading
  report.py      Markdown table + bar chart
data/            train / val / test questions and the passage corpus
results/         per-system predictions, metrics JSON, results table and chart
tests/           unit tests for the grader
```

## What LoRA is doing here

A full fine-tune updates all ~500M weights. LoRA freezes them and learns a
low-rank update for each targeted weight matrix: `W + (alpha/r) · B·A`, where
`A` is `r × d` and `B` is `d × r` with `r = 16`. Only `A` and `B` train, so the
adapter is a ~35 MB file that you can swap on and off the frozen base model.
`evaluate.py` does exactly that: it loads the base once and switches adapters with
`set_adapter` / `disable_adapter`.

Two details in `train_lora.py` matter:

- **Loss masking**: for Q&A examples the system prompt and question tokens get label
  `-100`, so the model is trained only to produce the answer, not to reproduce the prompt.
- **Passages as plain text**: passages are trained with ordinary next-token loss
  (continued pre-training), which is how new facts get into the weights.

## Notes and limitations

- **Lexical grading is approximate.** The grader can mark a correct paraphrase
  wrong, or a long answer that happens to contain the gold words right. Per-question
  predictions are saved in `results/predictions_*.jsonl` so you can audit them.
- **Yes/no questions are 44% of the set** and skew towards "Yes", so the report
  also shows an always-"Yes" baseline and splits accuracy by question type.
- Some questions only made sense next to their source article (e.g. *"What is
  the capital city?"*). Those are hard for every system.
- The Q&A-only adapter's validation loss rises after epoch 1 (1.89 → 2.24 by epoch 3).
  It overfits the 643 training answers, a reminder that more epochs aren't free.
- On a memory-constrained Mac, MPS memory fragments when every batch has a different
  shape. The trainer pads batches to multiples of 64 tokens and empties the MPS
  cache every 10 steps to stop the process from growing until the machine swaps.
