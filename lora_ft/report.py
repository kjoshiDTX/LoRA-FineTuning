"""Turn results/metrics_<split>.json into a Markdown table and a bar chart.

Usage: python -m lora_ft.report --split test
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lora_ft.config import DATA_DIR, RESULTS_DIR
from lora_ft.data import read_jsonl

LABELS = {
    "base": "Base model (closed book)",
    "lora_qa": "LoRA: Q&A only",
    "lora_qa_passages": "LoRA: Q&A + passages",
    "rag": "RAG (base + retrieval)",
    "lora_qa_passages+rag": "LoRA + RAG",
}
ACCURACY_COLOR = "#2a78d6"
HALLUCINATION_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"


def pct(x: float) -> str:
    return f"{x:.1%}"


def table(metrics: dict, split: str) -> str:
    questions = read_jsonl(DATA_DIR / f"{split}.jsonl")
    yes_no = [q for q in questions if q["yes_no"]]
    always_yes = sum(q["answer"] == "Yes" for q in yes_no) / len(yes_no)

    lines = [
        "| System | Accuracy | Hallucination rate | Abstention rate | Yes/no acc. | Factoid acc. | Factoid halluc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system, m in metrics.items():
        o, yn, fa = m["overall"], m["yes_no"], m["factoid"]
        lines.append(
            f"| {LABELS.get(system, system)} | {pct(o['accuracy'])} | {pct(o['hallucination_rate'])} "
            f"| {pct(o['abstention_rate'])} | {pct(yn['accuracy'])} | {pct(fa['accuracy'])} "
            f"| {pct(fa['hallucination_rate'])} |"
        )
    n = next(iter(metrics.values()))["overall"]["n"]
    lines.append("")
    lines.append(
        f"{n} held-out questions ({len(yes_no)} yes/no, {n - len(yes_no)} factoid). "
        f"For reference, answering \"Yes\" to every yes/no question scores {pct(always_yes)} on that subset."
    )
    return "\n".join(lines)


def chart(metrics: dict, path) -> None:
    systems = list(metrics)
    names = [LABELS.get(s, s) for s in systems]
    acc = [metrics[s]["overall"]["accuracy"] * 100 for s in systems]
    hal = [metrics[s]["overall"]["hallucination_rate"] * 100 for s in systems]

    fig, ax = plt.subplots(figsize=(8, 0.75 * len(systems) + 1.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = range(len(systems))
    h = 0.36
    bars_a = ax.barh([y - h / 2 - 0.01 for y in ys], acc, h, color=ACCURACY_COLOR, label="Accuracy")
    bars_h = ax.barh([y + h / 2 + 0.01 for y in ys], hal, h, color=HALLUCINATION_COLOR, label="Hallucination rate")
    for bars in (bars_a, bars_h):
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=TEXT)

    ax.set_yticks(list(ys), names, fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of held-out questions", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT)
    ax.set_title("Fine-tuning vs RAG on held-out questions", loc="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    metrics = json.loads((RESULTS_DIR / f"metrics_{args.split}.json").read_text())
    md = table(metrics, args.split)
    (RESULTS_DIR / f"results_{args.split}.md").write_text(md + "\n")
    chart(metrics, RESULTS_DIR / f"results_{args.split}.png")
    print(md)


if __name__ == "__main__":
    main()
