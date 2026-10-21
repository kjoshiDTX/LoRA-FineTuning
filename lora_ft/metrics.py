"""Grading for short-answer QA.

Every prediction gets exactly one label:
  correct      - matches the gold answer
  abstained    - the model said it does not know
  hallucinated - the model gave an answer, and it is wrong

Accuracy = correct / total. Hallucination rate = hallucinated / total.
The two do not sum to 1 because abstaining is neither right nor a hallucination.
"""

import re
import string
from collections import Counter

ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "i dont know",
    "not sure",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "does not contain",
    "doesn't contain",
    "not mentioned",
    "no information",
    "not provided",
)

F1_THRESHOLD = 0.5


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention(prediction: str) -> bool:
    text = prediction.lower().replace("’", "'")
    return not text.strip() or any(p in text for p in ABSTAIN_PATTERNS)


def yes_no_of(prediction: str) -> str | None:
    words = normalize(prediction).split()
    if words and words[0] in ("yes", "no"):
        return words[0].capitalize()
    return None


def grade(prediction: str, gold: str, yes_no: bool) -> str:
    if is_abstention(prediction):
        return "abstained"
    if yes_no:
        return "correct" if yes_no_of(prediction) == gold else "hallucinated"
    # Pad with spaces so containment is whole-word ("17" is not inside "1776").
    pred_norm, gold_norm = f" {normalize(prediction)} ", f" {normalize(gold)} "
    contained = pred_norm.strip() != "" and (gold_norm in pred_norm or pred_norm in gold_norm)
    if contained or token_f1(prediction, gold) >= F1_THRESHOLD:
        return "correct"
    return "hallucinated"


def summarize(rows: list[dict]) -> dict:
    """rows need 'label' and 'yes_no' keys; returns rates overall and by type."""

    def rates(subset: list[dict]) -> dict:
        n = len(subset)
        labels = Counter(r["label"] for r in subset)
        return {
            "n": n,
            "accuracy": labels["correct"] / n if n else 0.0,
            "hallucination_rate": labels["hallucinated"] / n if n else 0.0,
            "abstention_rate": labels["abstained"] / n if n else 0.0,
        }

    return {
        "overall": rates(rows),
        "yes_no": rates([r for r in rows if r["yes_no"]]),
        "factoid": rates([r for r in rows if not r["yes_no"]]),
    }
