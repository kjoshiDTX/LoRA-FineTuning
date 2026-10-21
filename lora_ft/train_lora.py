"""Fine-tune the base model with a LoRA adapter.

Two recipes, so you can see what fine-tuning does and does not teach:
  qa            - only the training Q&A pairs. Teaches answer *style*, but the
                  facts behind held-out questions are never shown to the model.
  qa_passages   - Q&A pairs plus every corpus passage as plain text. This is the
                  same knowledge RAG retrieves from, baked into the weights.

Usage: python -m lora_ft.train_lora --recipe qa_passages
"""

import argparse
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from lora_ft.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    CLOSED_BOOK_SYSTEM,
    DATA_DIR,
    SEED,
    device,
    dtype,
)
from lora_ft.data import read_jsonl

MAX_LEN = 256
PAD_MULTIPLE = 64  # few distinct batch shapes -> less allocator fragmentation
IGNORE = -100  # label value the loss function skips


def qa_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


def encode_qa(tokenizer, row: dict) -> dict:
    """Tokenize one chat example and mask the prompt, so the loss is computed
    only on the answer tokens."""
    messages = qa_messages(row)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
    labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels[: len(full_ids)]}


def encode_passage(tokenizer, passage: dict) -> dict:
    ids = tokenizer(passage["text"] + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ][:MAX_LEN]
    return {"input_ids": ids, "labels": list(ids)}


def build_examples(tokenizer, split: str, recipe: str) -> list[dict]:
    examples = [encode_qa(tokenizer, r) for r in read_jsonl(DATA_DIR / f"{split}.jsonl")]
    if recipe == "qa_passages" and split == "train":
        passages = read_jsonl(DATA_DIR / "passages.jsonl")
        examples += [encode_passage(tokenizer, p) for p in passages]
    random.Random(SEED).shuffle(examples)
    return examples


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        width = -(-width // PAD_MULTIPLE) * PAD_MULTIPLE
        input_ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


class EmptyCacheCallback(TrainerCallback):
    """Release cached GPU memory now and then. Without this, MPS memory grows
    with every new batch shape until the machine starts swapping."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=["qa", "qa_passages"], default="qa_passages")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="for quick smoke tests")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype()).to(device())
    if args.grad_checkpointing:
        # Trades compute for memory; only needed for larger models.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed for checkpointing with frozen base weights

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        # Adapt every linear layer in attention and MLP blocks.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train = build_examples(tokenizer, "train", args.recipe)
    val = build_examples(tokenizer, "val", args.recipe)
    print(f"train examples: {len(train)}  val examples: {len(val)}")

    out_dir = ADAPTER_DIR / args.recipe
    per_device = min(4, args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=max(1, args.batch_size // per_device),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # a float below 1 is a fraction of total steps
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[EmptyCacheCallback()],
    )

    start = time.time()
    trainer.train()
    print(f"training took {(time.time() - start) / 60:.1f} min")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    trainer.save_state()
    print(f"saved adapter to {out_dir}")


if __name__ == "__main__":
    main()
