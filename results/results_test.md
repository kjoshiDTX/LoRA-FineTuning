| System | Accuracy | Hallucination rate | Abstention rate | Yes/no acc. | Factoid acc. | Factoid halluc. |
|---|---:|---:|---:|---:|---:|---:|
| Base model (closed book) | 23.0% | 72.1% | 4.9% | 39.5% | 9.8% | 81.4% |
| LoRA: Q&A only | 40.4% | 59.6% | 0.0% | 76.5% | 11.8% | 88.2% |
| LoRA: Q&A + passages | 43.2% | 56.8% | 0.0% | 80.2% | 13.7% | 86.3% |
| RAG (base + retrieval) | 63.9% | 36.1% | 0.0% | 77.8% | 52.9% | 47.1% |
| LoRA + RAG | 67.8% | 32.2% | 0.0% | 84.0% | 54.9% | 45.1% |

183 held-out questions (81 yes/no, 102 factoid). For reference, answering "Yes" to every yes/no question scores 80.2% on that subset.
