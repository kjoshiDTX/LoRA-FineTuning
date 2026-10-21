"""Answer the held-out questions with each system and grade the answers.

Systems:
  base                   - base model, closed book
  lora_qa                - LoRA trained on Q&A pairs only, closed book
  lora_qa_passages       - LoRA trained on Q&A pairs + corpus passages, closed book
  rag                    - base model + top-k retrieved passages
  lora_qa_passages+rag   - fine-tuned model + retrieved passages

Usage: python -m lora_ft.evaluate --split test
"""

import argparse
import json
import time
from contextlib import nullcontext

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_ft.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    CLOSED_BOOK_SYSTEM,
    DATA_DIR,
    RAG_SYSTEM,
    RESULTS_DIR,
    device,
    dtype,
)
from lora_ft.data import read_jsonl, write_jsonl
from lora_ft.metrics import grade, normalize, summarize
from lora_ft.rag import Retriever, format_context

ALL_SYSTEMS = ["base", "lora_qa", "lora_qa_passages", "rag", "lora_qa_passages+rag"]


def build_prompt(tokenizer, question: str, context: str | None) -> str:
    if context is None:
        messages = [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], batch_size: int) -> list[str]:
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True)
        batch = batch.to(model.device)
        tokens = model.generate(
            **batch,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = tokens[:, batch["input_ids"].shape[1] :]
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            outputs.append(text.strip().split("\n")[0].strip())
    return outputs


def load_model(systems: list[str]):
    """Load the base model once and attach every adapter we need by name, so
    systems can switch adapters without reloading weights."""
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype()).to(device())
    adapters = sorted({s.split("+")[0] for s in systems if s.startswith("lora")})
    for i, name in enumerate(adapters):
        path = ADAPTER_DIR / name.removeprefix("lora_")  # e.g. adapters/qa_passages
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--systems", nargs="+", default=ALL_SYSTEMS, choices=ALL_SYSTEMS)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = read_jsonl(DATA_DIR / f"{args.split}.jsonl")[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"  # decoder-only models generate from the right edge
    model = load_model(args.systems)

    contexts = None
    if any(s.endswith("rag") for s in args.systems):
        retriever = Retriever(top_k=args.top_k)
        retrieved = retriever.search([q["question"] for q in questions])
        contexts = [format_context(r) for r in retrieved]

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / f"metrics_{args.split}.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    for system in args.systems:
        adapter = system.split("+")[0]
        uses_rag = system.endswith("rag")
        if adapter.startswith("lora"):
            model.set_adapter(adapter)
            ctx = nullcontext()
        else:
            ctx = model.disable_adapter() if isinstance(model, PeftModel) else nullcontext()

        prompts = [
            build_prompt(tokenizer, q["question"], contexts[i] if uses_rag else None)
            for i, q in enumerate(questions)
        ]
        start = time.time()
        with ctx:
            predictions = generate(model, tokenizer, prompts, args.batch_size)
        elapsed = time.time() - start

        rows = []
        for i, (q, pred) in enumerate(zip(questions, predictions)):
            row = {**q, "prediction": pred, "label": grade(pred, q["answer"], q["yes_no"])}
            if uses_rag:
                # Retrieval diagnostic: is the gold answer literally in the context?
                row["answer_in_context"] = (
                    None if q["yes_no"] else normalize(q["answer"]) in normalize(contexts[i])
                )
            rows.append(row)
        write_jsonl(RESULTS_DIR / f"predictions_{args.split}_{system}.jsonl", rows)

        summary = summarize(rows)
        summary["seconds"] = round(elapsed, 1)
        metrics[system] = summary
        o = summary["overall"]
        print(
            f"{system:>22}: accuracy {o['accuracy']:.1%}  "
            f"hallucination {o['hallucination_rate']:.1%}  "
            f"abstain {o['abstention_rate']:.1%}  ({elapsed:.0f}s)"
        )

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
