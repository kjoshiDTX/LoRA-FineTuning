"""Download the mini-Wikipedia dataset, clean it, and write train/val/test splits.

Usage: python -m lora_ft.data
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from lora_ft.config import DATA_DIR, DATASET, SEED, TRAIN_FRAC, VAL_FRAC


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 (e.g. 'RÃ\xado' -> 'Río')."""
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_answer(answer: str) -> str:
    answer = " ".join(answer.split()).strip()
    if answer.lower().rstrip(".") in ("yes", "no"):
        return answer.lower().rstrip(".").capitalize()
    return answer


def is_yes_no(answer: str) -> bool:
    return answer in ("Yes", "No")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def split_questions(rows: list[dict]) -> dict[str, list[dict]]:
    """Stratified split so yes/no and factoid questions appear in each split
    in the same proportion."""
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for group in (True, False):
        bucket = [r for r in rows if r["yes_no"] is group]
        rng.shuffle(bucket)
        n_train = round(len(bucket) * TRAIN_FRAC)
        n_val = round(len(bucket) * VAL_FRAC)
        splits["train"] += bucket[:n_train]
        splits["val"] += bucket[n_train : n_train + n_val]
        splits["test"] += bucket[n_train + n_val :]
    for split in splits.values():
        split.sort(key=lambda r: r["id"])
    return splits


def main() -> None:
    qa = load_dataset(DATASET, "question-answer")["test"]
    corpus = load_dataset(DATASET, "text-corpus")["passages"]

    questions = []
    for row in qa:
        answer = clean_answer(fix_mojibake(row["answer"]))
        questions.append(
            {
                "id": row["id"],
                "question": " ".join(fix_mojibake(row["question"]).split()),
                "answer": answer,
                "yes_no": is_yes_no(answer),
            }
        )

    passages = []
    for row in corpus:
        text = " ".join(fix_mojibake(row["passage"]).split())
        if len(text.split()) >= 5:
            passages.append({"id": row["id"], "text": text})

    splits = split_questions(questions)
    for name, rows in splits.items():
        write_jsonl(DATA_DIR / f"{name}.jsonl", rows)
    write_jsonl(DATA_DIR / "passages.jsonl", passages)

    for name, rows in splits.items():
        n_yes_no = sum(r["yes_no"] for r in rows)
        print(f"{name:>5}: {len(rows):4d} questions ({n_yes_no} yes/no)")
    print(f"passages: {len(passages)}")


if __name__ == "__main__":
    main()
